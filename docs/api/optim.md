# Sequence design

```{eval-rst}
.. currentmodule:: blochsim
```

Minimization of a user-written cost over bounded sequence parameters; see
{doc}`../explanations/sequence-design`.

{class}`SequenceDesign` holds the cost and the parameters and runs the loop.
{class}`Bounded` keeps a parameter inside given limits. {func}`crlb` returns the
Cramer-Rao lower bound on parameter variance from a signal Jacobian.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   Bounded
   SequenceDesign
   crlb
```
