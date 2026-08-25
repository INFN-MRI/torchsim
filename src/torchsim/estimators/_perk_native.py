"""The fused PERK kernels on the host, over the C++ extension.

The extension takes data pointers rather than tensors, so everything it is
given has to be contiguous, float32, and alive for the call. That is what this
module is for.
"""

from __future__ import annotations

__all__ = ["regress", "regress_vjp"]

import torch

from .. import _perk_cpu

#: Let the extension's pool size itself against the work.
_THREADS = 0


def _ready(tensor: torch.Tensor) -> torch.Tensor:
    """A contiguous float32 view the extension can take the address of."""
    return tensor.detach().to(torch.float32).contiguous()


def regress(
    signals: torch.Tensor,
    frequency: torch.Tensor,
    transposed: torch.Tensor,
    phase: torch.Tensor,
    feature_mean: torch.Tensor,
    weight: torch.Tensor,
    parameter_mean: torch.Tensor,
) -> torch.Tensor:
    """Estimate parameters from ``(voxels, contrasts)`` signals.

    ``transposed`` is ``frequency`` with its axes swapped and made contiguous,
    which is the layout the kernel's inner loop reads along.
    """
    held = [
        _ready(tensor)
        for tensor in (
            signals, frequency, transposed, phase, feature_mean, weight,
            parameter_mean,
        )
    ]
    voxels, contrasts = held[0].shape
    output = torch.empty(
        (voxels, weight.shape[0]), dtype=torch.float32, device="cpu"
    )
    _perk_cpu.regress(
        tuple(tensor.data_ptr() for tensor in (*held, output)),
        voxels,
        contrasts,
        frequency.shape[0],
        weight.shape[0],
        _THREADS,
    )
    return output


def regress_vjp(
    cotangent: torch.Tensor,
    signals: torch.Tensor,
    frequency: torch.Tensor,
    transposed: torch.Tensor,
    phase: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """The derivative of :func:`regress` with respect to ``signals``."""
    held = [
        _ready(tensor)
        for tensor in (signals, frequency, transposed, phase, weight, cotangent)
    ]
    voxels, contrasts = held[0].shape
    output = torch.empty_like(held[0])
    _perk_cpu.regress_vjp(
        tuple(tensor.data_ptr() for tensor in (*held, output)),
        voxels,
        contrasts,
        frequency.shape[0],
        weight.shape[0],
        _THREADS,
    )
    return output
