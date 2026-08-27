Framework
---------

What TorchSim is made of, from calling a sequence to writing one.

Start with a simulator that already exists: name its protocol, hand it tissue,
and read back the signal and its derivatives -- with respect to the tissue,
which is what a fit and a reconstruction descend, and with respect to the
schedule, which is what designs a protocol.

Then write your own. A **signal model** is two pieces: a state machine saying
what each kind of event does to the spins, and a simulator saying what order
they are played in. An **operator** is one module of a sequence -- a
preparation, a readout, a delay -- and writing one is writing a Python function
that returns events. Neither requires touching a kernel.
