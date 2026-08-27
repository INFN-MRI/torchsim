Model-based reconstruction
==========================

.. currentmodule:: torchsim

Solving for parameter maps straight from k-space, with the signal model inside
the forward operator rather than applied to images someone else reconstructed.

:class:`ModelOperator` is that model as an operator: parameter maps in, one
image per contrast out, with an analytic derivative that never builds a
Jacobian, a complex amplitude for proton density and receive phase, and the
same box bounds :class:`NonlinearLeastSquares` takes. It honours
:func:`execution`, and ``physics()`` hands it to deepinv.

:class:`GaussNewton` inverts the chain by repeated linearization. Which damping
it carries decides which method it is -- :class:`Schedule` for an iteratively
regularized Gauss-Newton, :class:`TrustRegion` for Levenberg-Marquardt, which
is what :class:`NonlinearLeastSquares` runs. How the linearized problem is
solved is a callable, and mostly it is somebody else's: :func:`iterative` hands
the linearized problem to deepinv's ``least_squares``, which minimizes exactly
what a Gauss-Newton step leaves. There is no conjugate gradient written here.
:func:`direct` is the exception and is not a general solver -- it is the batched
damped least-squares over a voxel-diagonal Jacobian that *is* the
Levenberg-Marquardt step. A closure around a proximal solver from elsewhere is
how a regularizer enters.

The Fourier encoding is not here and never will be. Anything exposing ``A`` and
``A_adjoint`` composes -- an mri-nufft operator through its deepinv bridge, say
-- and :attr:`Subspace.modes` hands the temporal basis to a subspace operator in
the layout it reads.

The operator
------------

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ModelOperator

The loop
--------

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   GaussNewton
   Schedule
   TrustRegion
   Linearization
   Solution

Linear solvers
--------------

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   direct
   iterative
