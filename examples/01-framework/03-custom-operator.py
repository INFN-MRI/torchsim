"""
======================
Writing a new operator
======================

An operator is one module of a sequence -- a pulse, a Readout, a Delay, a whole
preparation -- that knows what it plays and how long it holds the timeline.
Writing a new one is writing a Python function that returns events, and it
reaches the fused kernels with no change to them: no Triton, no C++.

This example builds a T2 preparation, checks it against the decay it is
supposed to impose, and registers it under a name so a stream that arrives
already labelled can ask for it.
"""

# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install torchsim

# %%
#
# We begin with the necessary imports:
#
import matplotlib.pyplot as plt
import torch

from torchsim.sequence import (
    compose,
    Delay,
    EpgEngine,
    EventAction,
    Excitation,
    ideal_rf_definition,
    module,
    operator,
    Readout,
    Refocusing,
    register_operator,
    SequenceDescription,
    TissueProperties,
)

# %%
# Composing one out of the ones that exist
# ----------------------------------------
# A T2 preparation tips the magnetization into the transverse plane, lets it
# decay for a chosen time about a Refocusing pulse, tips what is left back
# along z, and spoils whatever did not come back.
#
# All of those are operators already, so the preparation is
# :func:`~torchsim.sequence.module` over them. Nothing new is being taught to
# the kernels -- what is new is the *arrangement*, and that is exactly what an
# operator is.
#
# The Refocusing pulse is asked for uncrushed: a T2 preparation refocuses
# rather than dephases, and the crusher pair
# :func:`~torchsim.sequence.refocusing` adds by default would Spoil the echo it
# exists to form.


def t2_preparation(echo_time_s, *, spoil_s=2e-3):
    """Return a T2 preparation that weights the magnetization by its own decay.

    Parameters
    ----------
    echo_time_s:
        How long the magnetization spends in the transverse plane.
    spoil_s:
        The spoiler after the tip-up, which removes what did not return.
    """
    half = 0.5 * echo_time_s
    return module(
        Excitation(0.5 * torch.pi),
        Delay(half),
        Refocusing(torch.pi, 0.5 * torch.pi, crushed=False),
        Delay(half),
        Excitation(-0.5 * torch.pi),
        Delay(spoil_s, action=EventAction.SPOIL_AFTER),
        duration_s=echo_time_s + spoil_s,
    )


# %%
# Using it
# --------
# The preparation goes at the front of an ordinary refocused train, and
# :func:`~torchsim.sequence.compose` lays the two out end to end. The
# preparation leaves the weighted magnetization along z, so the train excites
# it as it would any other longitudinal magnetization.


def prepared_train(prep_s, echo_spacing_s, echoes):
    """Return a T2-prepared spin-echo train."""
    modules = [t2_preparation(prep_s), Excitation(0.5 * torch.pi, 0.5 * torch.pi)]
    for _ in range(echoes):
        modules.append(Delay(0.5 * echo_spacing_s))
        modules.append(Refocusing(torch.pi, 0.5 * torch.pi))
        modules.append(Delay(0.5 * echo_spacing_s))
        modules.append(Readout(0.5 * torch.pi))
    events, duration_s = compose(*modules)
    return SequenceDescription(
        subsequence_index=0,
        tr_duration_us=1e6 * duration_s,
        events=events,
        rf_definitions={0: ideal_rf_definition()},
    )


# %%
# Does it do what it says?
# ------------------------
# A preparation is worth only as much as the weighting it imposes, so we check
# it rather than assert it: sweep the preparation time and hold the first
# recorded echo against ``exp(-TE / T2)``, which is what a T2 preparation is
# for.
#
T2_MS = torch.tensor([40.0, 80.0, 160.0])
tissue = TissueProperties(t1_ms=1000.0, t2_ms=T2_MS)
prep_times_ms = torch.linspace(0.0, 120.0, 13)

prepared = []
for prep_ms in prep_times_ms:
    described = prepared_train(float(prep_ms) * 1e-3, 5e-3, echoes=1)
    result = EpgEngine().simulate(described, tissue, nstates=8)
    prepared.append(result.signal[..., 0].abs())

prepared = torch.stack(prepared, dim=-1)
weighting = prepared / prepared[:, :1]
expected = torch.exp(-prep_times_ms / T2_MS[:, None])

plt.figure()
plt.plot(prep_times_ms, weighting.T, "o")
plt.plot(prep_times_ms, expected.T, "-")
plt.xlabel("preparation time [ms]")
plt.ylabel("relative echo amplitude")
plt.legend(["T2 = 40 ms", "T2 = 80 ms", "T2 = 160 ms"])

print(
    "worst departure from exp(-TE/T2):",
    float((weighting - expected).abs().max()),
)

# %%
# The two follow each other to about 0.2%, and the residual is physics rather
# than error: the tipped-up magnetization recovers a little across the spoiler
# that follows it, by more for the longer preparations that leave less behind.

# %%
# Reaching it by name
# -------------------
# Events form a stream, and a stream can come from somewhere other than a
# builder -- an MRD file, a protocol exporter, any generator that names what it
# plays. Registering the operator is what lets such a stream ask for it without
# the caller keeping a mapping of its own.
#
register_operator("t2-prep", t2_preparation)

built = operator("t2-prep")(60e-3)
print("registered:", built.duration_s, "s,", len(built.emit(0.0)), "events")

# %%
# What an operator cannot say
# ---------------------------
# The vocabulary is the three event types, the four dephasing actions and the
# RF and ADC roles -- so a preparation, a Readout, a shaped or per-channel
# pulse is written here and reaches the kernels unchanged.
#
# What a description cannot express is *how much* a gradient dephases. It
# carries one crusher moment for the whole sequence and dephasing is quantized
# to whole configuration orders, so a bipolar pair, a velocity-encoding moment
# of its own, or a crusher of twice its neighbour's area have no spelling here.
# Those need a per-event gradient moment through the packed layout and every
# kernel, which is a change to the engine rather than to a module written on
# top of it.
