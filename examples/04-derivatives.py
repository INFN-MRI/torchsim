"""
===================================
Optimizing a clinical FSE echo train
===================================

Forward-mode derivatives of the nonlinear signal model are differentiated
again in reverse mode to optimize the refocusing train. Flip angles are
bounded by construction, while smoothness and RF-power terms keep the result
scanner-friendly.
"""

# %%
import matplotlib.pyplot as plt
import torch

from torchsim import FSET2Precision, SequenceOptimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t1_ms = torch.tensor([800.0, 1400.0], device=device)
t2_ms = torch.tensor([45.0, 120.0], device=device)
echo_spacing_ms = 5.0
echo_train_length = 16

# The package owns the constraints, objective, and optimization loop. Override
# ``objective`` in a subclass to study another acquisition criterion.
objective = FSET2Precision(
    t1_ms,
    t2_ms,
    echo_spacing_ms,
)
optimizer = SequenceOptimizer(
    objective,
    bounds=(30.0, 180.0),
    iterations=10,
)
result = optimizer.optimize(torch.full((echo_train_length,), 120.0, device=device))

# %%
optimized_flip = result.parameters.cpu()
figure, axes = plt.subplots(1, 2, figsize=(9, 3))
axes[0].plot(optimized_flip, ".-")
axes[0].set(xlabel="Echo", ylabel="Refocusing flip [deg]")
axes[1].semilogy(result.loss.cpu())
axes[1].set(xlabel="Iteration", ylabel="Constrained CRLB objective")
figure.tight_layout()
