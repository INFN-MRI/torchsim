Framework
---------

What TorchSim is made of, from calling a sequence to writing one.

Start with a simulator that already exists: name its protocol, hand it tissue,
and read back the signal and its derivatives -- with respect to the tissue,
which is what a fit and a reconstruction descend, and with respect to the
schedule, which is what designs a protocol.

Then go down a level at a time. A simulator can be given physics it does not
carry, which is a declaration rather than code. A sequence can be assembled
from operators that already exist, which is a list. A **signal model** of your
own is two pieces -- a state machine saying what each kind of event does to the
spins, and a simulator saying what order they are played in. And an
**operator** of your own is a Python function that returns events.

None of it requires touching a kernel.
