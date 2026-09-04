"""The linear route: a basis out, coefficient maps back in."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchsim import (
    Subspace,
)
from torchsim.estimators import DictionaryMatcher
from torchsim.simulators import MultiEchoSimulator

mrinufft = pytest.importorskip("mrinufft")

TE_MS = torch.linspace(10.0, 200.0, 8)
SIZE = 16
RANK = 3


@pytest.fixture(scope="module")
def basis():
    """A temporal basis for a multi-echo decay, from the model itself."""
    acquisition = MultiEchoSimulator(TE=TE_MS)
    signals = torch.as_tensor(
        acquisition.simulate(T2=torch.linspace(20.0, 300.0, 64))
    ).to(torch.complex64)
    return Subspace.fit(signals, RANK)


@pytest.fixture(scope="module")
def cartesian():
    """A fully sampled Cartesian trajectory, so the transform is exact."""
    if not mrinufft.check_backend("finufft"):
        pytest.skip("the finufft backend is unavailable")
    axis = np.arange(SIZE) - SIZE // 2
    x, y = np.meshgrid(axis, axis, indexing="ij")
    return np.stack([x.ravel(), y.ravel()], -1).astype(np.float32) / SIZE


def _operators(basis, cartesian):
    """A subspace operator and the single-contrast one to check it against."""
    from mrinufft.operators.subspace import MRISubspace

    build = mrinufft.get_operator("finufft")
    contrasts = TE_MS.numel()
    stacked = build(
        np.tile(cartesian, (contrasts, 1)),
        (SIZE, SIZE),
        n_coils=1,
        squeeze_dims=False,
    )
    single = build(cartesian, (SIZE, SIZE), n_coils=1, squeeze_dims=False)
    return MRISubspace(stacked, basis.modes.numpy()), single


def test_the_basis_goes_out_in_the_layout_the_encoding_reads(basis, cartesian) -> None:
    """The rank axis first, and a plain transpose rather than a conjugate one.

    Getting the conjugate wrong here would not raise -- it would reconstruct
    something plausible and wrong -- so it is checked against the operator
    that consumes it rather than reasoned about.
    """
    subspace_op, single = _operators(basis, cartesian)
    coefficients = torch.randn(
        (1, RANK, SIZE, SIZE),
        generator=torch.Generator().manual_seed(0),
        dtype=torch.complex64,
    )

    theirs = subspace_op.op(coefficients.numpy())[0, :, 0, :]

    images = basis.expand(coefficients.movedim(1, -1))
    ours = np.stack(
        [
            single.op(images[0, ..., echo].numpy()[None, None])[0, 0]
            for echo in range(TE_MS.numel())
        ]
    )
    assert np.abs(theirs - ours).max() < 1e-5 * np.abs(ours).max()


def test_the_adjoint_is_our_projection(basis, cartesian) -> None:
    """Coming back, the encoding projects exactly the way the mapping does."""
    subspace_op, single = _operators(basis, cartesian)
    kspace = (
        np.random.default_rng(1)
        .standard_normal((1, TE_MS.numel(), 1, cartesian.shape[0]))
        .astype(np.complex64)
    )

    theirs = subspace_op.adj_op(kspace)[0][:, 0].transpose(1, 2, 0)

    each = np.stack(
        [single.adj_op(kspace[0, echo][None])[0, 0] for echo in range(TE_MS.numel())]
    )
    ours = basis.project(torch.as_tensor(each).permute(1, 2, 0)).numpy()
    assert np.abs(theirs - ours).max() < 1e-5 * np.abs(ours).max()


def test_one_mapping_supplies_the_basis_and_reads_what_comes_back() -> None:
    """The linear route end to end, with no basis kept by hand anywhere.

    The mapping fits the basis, the reconstruction is given it, and the
    coefficients it returns go straight back to the mapping -- which must not
    project them a second time.
    """
    acquisition = MultiEchoSimulator(TE=TE_MS)
    grid = torch.linspace(20.0, 300.0, 128)
    mapping = DictionaryMatcher(acquisition).fit(T2=grid, M0=1.0, rank=RANK, seed=0)
    truth = torch.tensor([[40.0, 95.0], [180.0, 260.0]])
    images = torch.as_tensor(acquisition.simulate(T2=truth)).to(torch.complex64)

    coefficients = mapping.subspace.project(images)
    maps = mapping.from_coefficients(coefficients)

    step = float(torch.diff(grid).max())
    assert float((maps["T2"] - truth).abs().max()) <= step
    torch.testing.assert_close(maps["T2"], mapping(images)["T2"])


def test_coefficients_of_the_wrong_rank_are_refused() -> None:
    """A silent mismatch would reconstruct the wrong tissue, not fail."""
    acquisition = MultiEchoSimulator(TE=TE_MS)
    mapping = DictionaryMatcher(acquisition).fit(
        T2=torch.linspace(20.0, 300.0, 64), rank=RANK, seed=0
    )

    with pytest.raises(ValueError, match="rank 3"):
        mapping.from_coefficients(torch.zeros(4, 5))


def test_a_mapping_with_no_basis_has_no_coefficients_to_read() -> None:
    """Asking for them is a mistake about the mapping, not about the data."""
    mapping = DictionaryMatcher(MultiEchoSimulator(TE=TE_MS)).fit(
        T2=torch.linspace(20.0, 300.0, 64), seed=0
    )

    with pytest.raises(RuntimeError, match="no subspace"):
        mapping.from_coefficients(torch.zeros(4, 3))
