Framework
---------

What TorchSim is made of, from calling a sequence to writing one.

Start with a simulator that already exists: name its protocol, hand it tissue,
give it the pulse the scanner actually plays, and read back the signal and its
derivatives -- with respect to the tissue, which is what a fit and a
reconstruction descend, and with respect to the schedule, which is what designs
a protocol.

Then ask it for more physics. A model declares which tissue properties it
exposes, and that declaration is what decides which terms the kernels evaluate:
the transmit array, off resonance, exchange, magnetization transfer, diffusion
and flow are each a name in a mapping.

Then go down a level at a time. A **signal model** of your own is two pieces --
a state machine saying what each kind of event does to the spins, and a
simulator saying what order they are played in. And an **operator** of your own
-- a preparation, a readout that samples twice where the shipped ones sample
once -- is a Python function that returns events.

None of it requires touching a kernel.
