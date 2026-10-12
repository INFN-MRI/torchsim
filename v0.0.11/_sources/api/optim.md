# Sequence design

```{eval-rst}
.. currentmodule:: blochsim
```

Minimization of a user-written cost over bounded sequence parameters; see
{doc}`../explanations/sequence-design`.

{class}`SequenceDesign` holds the cost and the parameters and runs the loop.
{class}`Bounded` keeps a parameter inside given limits. {func}`crlb` returns the
Cramer-Rao lower bound on parameter variance from a signal Jacobian.

| Object | Description |
| --- | --- |
| {obj}`~blochsim.Bounded` | An initial value and the interval a designed parameter stays in. |
| {obj}`~blochsim.SequenceDesign` | A cost, the parameters it is minimized over, and their limits. |
| {obj}`~blochsim.crlb` | Cramer-Rao lower bound on the variance of each parameter. |
