# Security Policy

## Supported versions

Security fixes are applied to the latest release on PyPI and to `main`.
Older releases are not patched.

## Reporting a vulnerability

Report privately, not as a public issue:

- open a draft advisory through
  [Security -> Report a vulnerability](https://github.com/FiRMLAB-Pisa/torchsim/security/advisories/new),
  which is the preferred route; or
- email **matteo.cencini@gmail.com** if you cannot use GitHub.

Please include what an attacker can do, the version and platform you saw it
on, and a minimal reproduction. You can expect an acknowledgement within a
week, an assessment within two, and credit in the advisory unless you ask
otherwise. Please give us a chance to release a fix before disclosing
publicly.

## Scope

TorchSim executes the code you give it: a sequence layout, a cost function and
a signal model are all Python that runs in your process, so a malicious
*script* is out of scope in the same way it is for NumPy or PyTorch.

In scope is anything that turns *data* into execution or into memory
corruption -- a sequence description, a pulse waveform, a phantom or a
dictionary read from a file, or values passed to the simulator, reaching the
C++ and Triton kernels. Those kernels index raw pointers, so an out-of-bounds
read or write reachable from ordinary arguments is a vulnerability and not
merely a bug.

## Not a vulnerability

A simulation that returns wrong physics, diverges, or raises is a
[bug report](https://github.com/FiRMLAB-Pisa/torchsim/issues/new/choose).
