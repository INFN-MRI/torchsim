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

| Object | Description |
| --- | --- |
| {obj}`~blochsim.execution` | Set where per-voxel work runs inside the block. |
| {obj}`~blochsim.offload` | Stream host-resident volumes through CUDA devices inside the block. |

## Utilities

| Object | Description |
| --- | --- |
| {obj}`~blochsim.utils.b1rms` | Root-mean-square B1 of an RF pulse, in T. |
