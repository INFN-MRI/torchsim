Framework
---------

How to use TorchSim, and how to extend it.

The first notebook covers basic usage: simulating a signal, taking its
derivatives, tuning the run, and reading a sequence back from the description
a scanner streams. The second covers the physics a simulator can carry beyond
T1 and T2.

The last two are for sequences TorchSim does not ship. A **signal model** of
your own is two pieces -- a physics saying what each kind of event does, and a
simulator saying what order they are played in. An **operator** of your own --
a preparation, a readout -- is a Python function that returns events.

None of it requires touching a kernel.
