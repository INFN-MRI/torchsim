# Sequences

```{eval-rst}
.. currentmodule:: torchsim
```

Device-agnostic description of an acquisition, shared by the interpreter, the
estimators and the sequence optimizer. A description is a list of events with
the timing and the tissue interaction each one carries; nothing in it names a
vendor or a device.

This page is the description and what it is assembled from. Running one is on
{doc}`execution`, and the temporal basis a run spans is on {doc}`recon`.

## Description

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   SequenceDescription
```

### Event vocabulary

What an event may be, and what it does when it is reached.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   EventType
   ShimDefinition
```

## Operators

The modules a description is assembled from, and the registry that reaches one
by name. Writing a new operator is writing a Python function that returns
events; it reaches the fused kernels with no change to them.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   Operator
   Excitation
   Refocusing
   Inversion
   Saturation
   Readout
   Delay
   Dephase
   Spoil
   bSSFPReadout
   SSFPFidReadout
   SSFPEchoReadout
   SPGRReadout
   FSEReadout
```

## Builders

Whole sequences, assembled from the operators above.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

```
