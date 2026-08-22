"""EPG States creation."""

import torch
from types import SimpleNamespace

from utils.epg import states_matrix


def test_states_matrix_default():
    device = torch.device("cpu")
    nstates = 5

    result = states_matrix(device, nstates)

    assert isinstance(result, SimpleNamespace), "Result should be a SimpleNamespace"
    assert result.Fplus.shape == (nstates, 1, 1), "Fplus shape mismatch"
    assert result.Fminus.shape == (nstates, 1, 1), "Fminus shape mismatch"
    assert result.Z.shape == (nstates, 1, 1), "Z shape mismatch"

    assert torch.all(result.Fplus == 0), "Fplus should be initialized to zeros"
    assert torch.all(result.Fminus == 0), "Fminus should be initialized to zeros"
    assert torch.all(result.Z[0] == 1), "Z[0] should be initialized to ones"
    assert torch.all(result.Z[1:] == 0), "Z[1:] should be initialized to zeros"


def test_states_matrix_non_default():
    device = torch.device("cpu")
    nstates = 10
    nlocs = 4
    ntrans_pools = 2
    nlong_pools = 3
    weight = torch.as_tensor((0.6, 0.2, 0.2), dtype=torch.float32, device=device)

    result = states_matrix(device, nstates, nlocs, ntrans_pools, nlong_pools, weight)

    assert isinstance(result, SimpleNamespace), "Result should be a SimpleNamespace"
    assert result.Fplus.shape == (nstates, nlocs, ntrans_pools), "Fplus shape mismatch"
    assert result.Fminus.shape == (
        nstates,
        nlocs,
        ntrans_pools,
    ), "Fminus shape mismatch"
    assert result.Z.shape == (nstates, nlocs, nlong_pools), "Z shape mismatch"

    assert result.Fplus.device == device, "Fplus device mismatch"
    assert result.Fminus.device == device, "Fminus device mismatch"
    assert result.Z.device == device, "Z device mismatch"

    assert torch.all(result.Fplus == 0), "Fplus should be initialized to zeros"
    assert torch.all(result.Fminus == 0), "Fminus should be initialized to zeros"
    assert torch.all(
        result.Z[0, :, 0] == 0.6
    ), "Z[0] for pool 0 should be initialized to 0.6"
    assert torch.all(
        result.Z[0, :, 1] == 0.2
    ), "Z[0] for pool 1 should be initialized to 0.2"
    assert torch.all(
        result.Z[0, :, 2] == 0.2
    ), "Z[0] for pool 2 should be initialized to 0.2"
    assert torch.all(result.Z[1:] == 0), "Z[1:] should be initialized to zeros"
