//
//  Gemma3Text+AllHiddenStates.swift
//  mlx-swift-lm
//
//  Encoder-style multi-layer hidden-state extraction for Gemma 3 text models.
//  Added to support using Gemma 3 as a frozen text ENCODER (e.g. Lightricks
//  LTX-2's text conditioning), which needs every layer's hidden state rather
//  than just generated tokens.
//
//  Upstream candidate: github.com/xocialize/mlx-swift-lm @ ltx/gemma-all-hidden-states.
//

import Foundation
import MLX
import MLXFast

extension Gemma3Model {
    /// Returns the embedding output plus each transformer layer's output —
    /// `numHiddenLayers + 1` states, each shaped `(B, T, hiddenSize)`.
    ///
    /// A SINGLE uniform `mask` is applied to every layer. This is the
    /// text-encoder use: the caller supplies a combined causal+padding mask and
    /// the per-layer sliding-window/global mask selection used by
    /// `callAsFunction(_:mask:cache:)` is intentionally bypassed.
    /// Throws `CancellationError` between layers when the surrounding task is cancelled —
    /// the per-layer `eval` already bounds each step, so a cancel (user quit / engine preempt)
    /// lands within ~one layer's compute instead of riding the whole 48-layer forward.
    public func allHiddenStates(
        _ inputs: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) throws -> [MLXArray] {
        var h = embedTokens(inputs)
        let scale = MLXArray(sqrt(Float(config.hiddenSize)), dtype: .bfloat16)
        h = h * scale.asType(h.dtype)

        var states: [MLXArray] = [h]
        for layer in layers {
            try Task.checkCancellation()
            h = layer(h, mask: mask, cache: nil)
            // Per-layer materialization keeps each Metal command buffer below the
            // macOS GPU watchdog (~10s) — without it, all layers fuse into one
            // dispatch and time out. Matches the oracle's LTX2_GEMMA_EVAL_EVERY=1.
            eval(h)
            states.append(h)
        }
        return states
    }
}

extension Gemma3TextModel {
    /// See ``Gemma3Model/allHiddenStates(_:mask:)``. Convenience forwarding from
    /// the top-level text model to its inner `model`.
    public func allHiddenStates(
        _ inputs: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) throws -> [MLXArray] {
        try model.allHiddenStates(inputs, mask: mask)
    }
}
