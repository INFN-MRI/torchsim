Parameter inference
-------------------

Estimating tissue properties from a measured volume.

Every estimator is made from the simulator it inverts and fitted over the
same statement of the problem -- what is unknown, over what range, at what
noise level -- and they differ only in how they fill it in. These examples change only that: dictionary matching over a parameter
grid, compressed and clustered; interpolation along a curve where there is a
single unknown; a nonlinear fit of the model itself; and a kernel regression
that never builds a grid at all.

Every one of them maps the same BrainWeb slice, so the answer is known
everywhere -- mixtures of tissues included -- and reports what it cost in time
and in memory alongside what it got wrong.
