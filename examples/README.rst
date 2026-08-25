.. _general_examples:

Examples
========

This is a collection of examples showing how to use TorchSim to create MR
simulators for different applications.

The gallery progresses from writing a signal model, to generating synthetic
data, to quantitative parameter mapping with dictionary matching and PERK, and
then to the shared linear-subspace and nonlinear-model workflow. One example
goes the other way and extends the sequence vocabulary itself, by writing a
new operator.

Two of them design a sequence rather than simulate one, and they are the same
three pieces with a different cost in the middle: one chooses the flip angles
of a joint relaxometry experiment so that T1 and T2 are estimated as precisely
as possible, the other chooses the refocusing trains of a segmented 3D turbo
spin echo so that the image is sharp where sharpness is decided and has
contrast where contrast is decided.
