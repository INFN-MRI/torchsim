# Execution

```{eval-rst}
.. currentmodule:: blochsim
```

Context managers that set where per-voxel work runs: on the host, on one
device, or streamed in chunks through several.

Under a policy, work too small to repay a device launch stays on the host,
work that fits is sent whole, and larger work is streamed in chunks. See
{doc}`../explanations/execution`.

## Placement

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   execution
   offload
```

## Utilities

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   utils.b1rms
```
