"""
=====================================
Linear subspace and nonlinear imaging
=====================================

The same SequenceDescription can generate a low-rank temporal basis for a
linear subspace reconstruction and act as the nonlinear signal model in a
model-based reconstruction or fitting problem.
"""

# %%
import matplotlib.pyplot as plt
import torch

from torchsim import EpgEngine, fse_description, simulate_subspace, TissueProperties

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
description = fse_description(
    torch.deg2rad(torch.full((40,), 150.0, device=device)),
    echo_spacing_s=5e-3,
    phases_rad=torch.pi / 2,
)

# %%
# Linear route: simulate the expected tissue range and retain its leading
# temporal singular vectors.
t2_dictionary = torch.linspace(20.0, 300.0, 512, device=device)
subspace = simulate_subspace(
    description,
    TissueProperties(t1_ms=1000.0, t2_ms=t2_dictionary),
    rank=5,
    nstates=12,
)
dictionary = subspace.dictionary
projected = (dictionary @ subspace.basis) @ subspace.basis.mH
relative_error = (dictionary - projected).norm() / dictionary.norm()
print(f"rank-5 relative dictionary error: {relative_error:.3e}")

# %%
# Nonlinear route: use the simulator directly and let Torch differentiate the
# forward model. In a model-based reconstruction this function is composed
# with coil sensitivities and Fourier encoding instead of the small loss here.
t2_true = torch.tensor(87.0, device=device)
measured = EpgEngine().simulate(
    description,
    TissueProperties(t1_ms=1000.0, t2_ms=t2_true),
    nstates=12,
).signal.detach()
raw_t2 = torch.nn.Parameter(torch.log(torch.tensor(60.0, device=device)))
optimizer = torch.optim.LBFGS([raw_t2], max_iter=25, line_search_fn="strong_wolfe")


def closure() -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    t2_ms = raw_t2.exp()
    predicted = EpgEngine().simulate(
        description,
        TissueProperties(t1_ms=1000.0, t2_ms=t2_ms),
        nstates=12,
    ).signal
    loss = (predicted - measured).abs().square().mean()
    loss.backward()
    return loss


optimizer.step(closure)
print(f"nonlinear estimate: {raw_t2.exp().item():.2f} ms")

plt.plot(dictionary[::64].abs().mT.cpu())
plt.xlabel("Echo")
plt.ylabel("Signal magnitude")
plt.tight_layout()

