# Sequence design

```{eval-rst}
.. currentmodule:: blochsim.optim
```

Minimization of a user-written cost over bounded sequence parameters; see
{doc}`../explanations/sequence-design`.

{class}`SequenceDesign` holds the cost and the parameters and runs the loop.
{class}`Bounded` keeps a parameter inside given limits. {func}`crlb` returns the
Cramer-Rao lower bound on parameter variance from a signal Jacobian.

| Object | Description |
| --- | --- |
| {obj}`~blochsim.optim.Bounded` | An initial value and the interval a designed parameter stays in. |
| {obj}`~blochsim.optim.SequenceDesign` | A cost, the parameters it is minimized over, and their limits. |
| {obj}`~blochsim.optim.crlb` | Cramer-Rao lower bound on the variance of each parameter. |
