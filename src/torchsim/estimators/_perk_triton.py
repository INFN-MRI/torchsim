"""Fused Triton kernels for the PERK feature map and its regression.

Estimating a parameter from a signal is, once PERK is fitted, one line of
arithmetic per voxel::

    y = parameter_mean + (scale * cos(W @ x + b) - feature_mean) @ weight.T

Written as Torch operations that line builds the whole ``(voxels, features)``
matrix, writes it to memory and reads it back to contract it away again. On a
million voxels at a thousand features that is nearly four gigabytes of traffic
carrying nothing the answer needs.

Here a tile of features is formed and consumed into the output accumulator
while it is still in registers, so the matrix never exists. The adjoint does
the same and recomputes the tile rather than keeping it, for the same reason.
"""

from __future__ import annotations

__all__ = ["regress", "regress_vjp"]

import math

import torch
import triton
import triton.language as tl

#: Tile shapes, swept on one card over voxel, feature and contrast blocks of
#: 32 to 128. The feature and contrast blocks are what ``tl.dot`` sees, so they
#: cannot go below the 16 it requires however narrow the problem is.
_BLOCK_VOXELS = 128
_BLOCK_FEATURES = 64
_BLOCK_CONTRASTS = 32
_WARPS = 8
_STAGES = 2
_MIN_DOT = 16

#: Both contractions run at full float32.
#:
#: The output is a sum of a thousand terms that cancel, so a relative error on
#: each shows up magnified in the sum: measured against a float64 reference,
#: TF32 on either contraction gives 5e-4 to 7e-4 where full float32 gives
#: 6.6e-7 -- which is what Torch's own float32 gives on the same expression.
#: TF32 would be three times faster and three orders of magnitude further from
#: the answer, so the kernel computes the same function the composed path does
#: and the speed comes from not writing the features to memory.
_PRECISION = "ieee"


@triton.jit
def _regress_kernel(
    signal_ptr,
    frequency_ptr,
    phase_ptr,
    feature_mean_ptr,
    weight_ptr,
    parameter_mean_ptr,
    output_ptr,
    voxels,
    contrasts,
    features,
    parameters,
    scale,
    stride_sv,
    stride_sc,
    stride_ff,
    stride_fc,
    stride_wp,
    stride_wf,
    stride_ov,
    stride_op,
    BLOCK_VOXELS: tl.constexpr,
    BLOCK_FEATURES: tl.constexpr,
    BLOCK_CONTRASTS: tl.constexpr,
    BLOCK_PARAMETERS: tl.constexpr,
    PRECISION: tl.constexpr,
):
    """One block of voxels, all the way from signal to parameters."""
    voxel = tl.program_id(0) * BLOCK_VOXELS + tl.arange(0, BLOCK_VOXELS)
    live = voxel < voxels
    parameter = tl.arange(0, BLOCK_PARAMETERS)
    wanted = parameter < parameters

    total = tl.zeros((BLOCK_VOXELS, BLOCK_PARAMETERS), dtype=tl.float32)
    for start in range(0, features, BLOCK_FEATURES):
        feature = start + tl.arange(0, BLOCK_FEATURES)
        present = feature < features
        angle = tl.zeros((BLOCK_VOXELS, BLOCK_FEATURES), dtype=tl.float32)
        for first in range(0, contrasts, BLOCK_CONTRASTS):
            contrast = first + tl.arange(0, BLOCK_CONTRASTS)
            here = contrast < contrasts
            block = tl.load(
                signal_ptr + voxel[:, None] * stride_sv + contrast[None, :] * stride_sc,
                mask=live[:, None] & here[None, :],
                other=0.0,
            )
            rows = tl.load(
                frequency_ptr
                + feature[:, None] * stride_ff
                + contrast[None, :] * stride_fc,
                mask=present[:, None] & here[None, :],
                other=0.0,
            )
            angle += tl.dot(block, tl.trans(rows), input_precision=PRECISION)
        shift = tl.load(phase_ptr + feature, mask=present, other=0.0)
        centre = tl.load(feature_mean_ptr + feature, mask=present, other=0.0)
        mapped = scale * tl.cos(angle + shift[None, :]) - centre[None, :]
        mapped = tl.where(present[None, :], mapped, 0.0)
        columns = tl.load(
            weight_ptr + parameter[:, None] * stride_wp + feature[None, :] * stride_wf,
            mask=wanted[:, None] & present[None, :],
            other=0.0,
        )
        total += tl.dot(mapped, tl.trans(columns), input_precision=PRECISION)

    offset = tl.load(parameter_mean_ptr + parameter, mask=wanted, other=0.0)
    tl.store(
        output_ptr + voxel[:, None] * stride_ov + parameter[None, :] * stride_op,
        total + offset[None, :],
        mask=live[:, None] & wanted[None, :],
    )


@triton.jit
def _regress_vjp_kernel(
    signal_ptr,
    frequency_ptr,
    phase_ptr,
    weight_ptr,
    cotangent_ptr,
    output_ptr,
    voxels,
    contrasts,
    features,
    parameters,
    scale,
    stride_sv,
    stride_sc,
    stride_ff,
    stride_fc,
    stride_wp,
    stride_wf,
    stride_cv,
    stride_cp,
    stride_ov,
    stride_oc,
    BLOCK_VOXELS: tl.constexpr,
    BLOCK_FEATURES: tl.constexpr,
    BLOCK_CONTRASTS: tl.constexpr,
    BLOCK_PARAMETERS: tl.constexpr,
    PRECISION: tl.constexpr,
):
    """The derivative of one block of voxels with respect to their signals.

    The angle the forward pass formed is rebuilt rather than stored: a tile is
    cheaper to compute twice than to carry through memory once, which is the
    same trade that makes the forward pass worth fusing.
    """
    voxel = tl.program_id(0) * BLOCK_VOXELS + tl.arange(0, BLOCK_VOXELS)
    live = voxel < voxels
    parameter = tl.arange(0, BLOCK_PARAMETERS)
    wanted = parameter < parameters

    seed = tl.load(
        cotangent_ptr + voxel[:, None] * stride_cv + parameter[None, :] * stride_cp,
        mask=live[:, None] & wanted[None, :],
        other=0.0,
    )

    for outer in range(0, contrasts, BLOCK_CONTRASTS):
        column = outer + tl.arange(0, BLOCK_CONTRASTS)
        writing = column < contrasts
        gradient = tl.zeros((BLOCK_VOXELS, BLOCK_CONTRASTS), dtype=tl.float32)
        for start in range(0, features, BLOCK_FEATURES):
            feature = start + tl.arange(0, BLOCK_FEATURES)
            present = feature < features
            angle = tl.zeros((BLOCK_VOXELS, BLOCK_FEATURES), dtype=tl.float32)
            for first in range(0, contrasts, BLOCK_CONTRASTS):
                contrast = first + tl.arange(0, BLOCK_CONTRASTS)
                here = contrast < contrasts
                block = tl.load(
                    signal_ptr
                    + voxel[:, None] * stride_sv
                    + contrast[None, :] * stride_sc,
                    mask=live[:, None] & here[None, :],
                    other=0.0,
                )
                rows = tl.load(
                    frequency_ptr
                    + feature[:, None] * stride_ff
                    + contrast[None, :] * stride_fc,
                    mask=present[:, None] & here[None, :],
                    other=0.0,
                )
                angle += tl.dot(block, tl.trans(rows), input_precision=PRECISION)
            shift = tl.load(phase_ptr + feature, mask=present, other=0.0)
            columns = tl.load(
                weight_ptr
                + parameter[:, None] * stride_wp
                + feature[None, :] * stride_wf,
                mask=wanted[:, None] & present[None, :],
                other=0.0,
            )
            through = tl.dot(seed, columns, input_precision=PRECISION)
            through *= -scale * tl.sin(angle + shift[None, :])
            through = tl.where(present[None, :], through, 0.0)
            rows = tl.load(
                frequency_ptr
                + feature[:, None] * stride_ff
                + column[None, :] * stride_fc,
                mask=present[:, None] & writing[None, :],
                other=0.0,
            )
            gradient += tl.dot(through, rows, input_precision=PRECISION)
        tl.store(
            output_ptr + voxel[:, None] * stride_ov + column[None, :] * stride_oc,
            gradient,
            mask=live[:, None] & writing[None, :],
        )


def _blocks(signals: torch.Tensor, parameters: int) -> dict[str, int]:
    """Tile shapes for one problem, never below what ``tl.dot`` accepts."""
    contrasts = signals.shape[-1]
    return {
        "BLOCK_VOXELS": _BLOCK_VOXELS,
        "BLOCK_FEATURES": _BLOCK_FEATURES,
        "BLOCK_CONTRASTS": max(
            _MIN_DOT, min(_BLOCK_CONTRASTS, triton.next_power_of_2(contrasts))
        ),
        "BLOCK_PARAMETERS": max(_MIN_DOT, triton.next_power_of_2(parameters)),
        "PRECISION": _PRECISION,
        "num_warps": _WARPS,
        "num_stages": _STAGES,
    }


def regress(
    signals: torch.Tensor,
    frequency: torch.Tensor,
    phase: torch.Tensor,
    feature_mean: torch.Tensor,
    weight: torch.Tensor,
    parameter_mean: torch.Tensor,
) -> torch.Tensor:
    """Estimate parameters from ``(voxels, contrasts)`` signals.

    Returns
    -------
    torch.Tensor
        ``(voxels, parameters)``.
    """
    signals = signals.contiguous()
    voxels, contrasts = signals.shape
    features = frequency.shape[0]
    parameters = weight.shape[0]
    output = torch.empty(
        (voxels, parameters), dtype=torch.float32, device=signals.device
    )
    grid = (triton.cdiv(voxels, _BLOCK_VOXELS),)
    _regress_kernel[grid](
        signals,
        frequency,
        phase,
        feature_mean,
        weight,
        parameter_mean,
        output,
        voxels,
        contrasts,
        features,
        parameters,
        math.sqrt(2.0 / features),
        signals.stride(0),
        signals.stride(1),
        frequency.stride(0),
        frequency.stride(1),
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
        **_blocks(signals, parameters),
    )
    return output


def regress_vjp(
    cotangent: torch.Tensor,
    signals: torch.Tensor,
    frequency: torch.Tensor,
    phase: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """The derivative of :func:`regress` with respect to ``signals``.

    Returns
    -------
    torch.Tensor
        ``(voxels, contrasts)``.
    """
    signals = signals.contiguous()
    cotangent = cotangent.contiguous()
    voxels, contrasts = signals.shape
    features = frequency.shape[0]
    parameters = weight.shape[0]
    output = torch.empty_like(signals)
    grid = (triton.cdiv(voxels, _BLOCK_VOXELS),)
    _regress_vjp_kernel[grid](
        signals,
        frequency,
        phase,
        weight,
        cotangent,
        output,
        voxels,
        contrasts,
        features,
        parameters,
        math.sqrt(2.0 / features),
        signals.stride(0),
        signals.stride(1),
        frequency.stride(0),
        frequency.stride(1),
        weight.stride(0),
        weight.stride(1),
        cotangent.stride(0),
        cotangent.stride(1),
        output.stride(0),
        output.stride(1),
        **_blocks(signals, parameters),
    )
    return output
