Sequence optimization
---------------------

Choosing a sequence's parameters rather than simulating the ones you were
given.

A design problem is three pieces: a simulator with the tissue it is designed
for already fixed on it; a cost, which is a plain function of what that
simulator records; and a :class:`~torchsim.SequenceDesign`, which holds the
parameters inside the limits the scanner will play and runs the loop.

Only the cost changes between the two examples here. One asks for a picture --
sharp where sharpness is decided, with contrast where contrast is decided --
and the other asks for precision, choosing flip angles so that T1 and T2 are
estimated as tightly as the scan time allows.
