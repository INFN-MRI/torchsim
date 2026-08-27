.. _general_examples:

Examples
========

Worked examples, grouped by what you are trying to do.

**Framework** is the vocabulary: run a sequence that ships with TorchSim, take
its derivatives, then write one of your own -- a signal model, and the
operators a sequence is assembled from.

**Parameter inference** turns a measured volume into maps. The same problem is
stated once and handed to a different estimator each time, always on the same
BrainWeb slice, so that what each costs and what each gets wrong are read off
the same numbers.

**Sequence optimization** goes the other way and chooses the sequence. The
three pieces are always the same -- an acquisition, a cost, a bounded set of
parameters -- and only the cost tells a precision design from an image-quality
one.

**Model-based imaging** reconstructs the maps straight from k-space, with the
signal model inside the forward operator, by a linear subspace or by nonlinear
inversion.

**Miscellaneous** collects everything else.
