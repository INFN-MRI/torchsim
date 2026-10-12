# Internals

Derivations, format rules and kernels behind {doc}`../../explanations/index`,
for whoever changes them.

| Page | Scope |
| --- | --- |
| {doc}`epg` | The extended phase graph operators the kernels implement, and how each tissue term enters. |
| {doc}`description` | How a Pulseq file and an MRD stream are read into a sequence description. |
| {doc}`kernels` | How a sequence description becomes packed buffers, and how the fused CPU and CUDA kernels run them. |

```{toctree}
:hidden:

epg
description
kernels
```
