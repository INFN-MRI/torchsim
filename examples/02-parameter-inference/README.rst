Parameter inference
-------------------

Estimating tissue properties from a measured volume.

A :class:`~torchsim.ParameterMapping` states the problem -- what is unknown,
over what range, from what acquisition, at what noise level -- and the method
that fills it in is a separate choice. These examples change only that choice:
dictionary matching over a parameter grid, compressed and clustered;
interpolation along a curve where there is a single unknown; a nonlinear fit of
the model itself; and a kernel regression that never builds a grid at all.

Every one of them maps the same BrainWeb slice, so the answer is known
everywhere -- mixtures of tissues included -- and reports what it cost in time
and in memory alongside what it got wrong.
