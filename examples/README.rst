.. _general_examples:

Examples
========

This is a collection of examples showing how to use TorchSim to create MR
simulators for different applications.

The gallery progresses from writing a signal model, to generating synthetic
data, to mapping T1, T2 and proton density together from an MR fingerprinting
train -- by exhaustive matching, by matching a clustered dictionary, and by a
kernel regression, all three working in the low-rank basis the train actually
spans -- and then to the shared linear-subspace and nonlinear-model workflow.
One example goes the other way and extends the sequence vocabulary itself, by
writing a new operator.

The last one goes further and never forms the contrast images at all: it
reconstructs a T2 map straight from undersampled radial k-space, with the
signal model inside the forward operator and the Fourier encoding supplied by
mri-nufft. Gridding, a linear subspace and the nonlinear model are run on the
same data so what each route costs and gets wrong can be read off a table.

Two of them design a sequence rather than simulate one, and they are the same
three pieces with a different cost in the middle: one chooses the flip angles
of a joint relaxometry experiment so that T1 and T2 are estimated as precisely
as possible, the other chooses the refocusing trains of a segmented 3D turbo
spin echo so that the image is sharp where sharpness is decided and has
contrast where contrast is decided.
