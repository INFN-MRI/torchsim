"""RF pulse operators."""

__all__ = [
    "rf_pulse_op",
    "phased_rf_pulse_op",
    "spinor_rf_pulse_op",
    "multidrive_rf_pulse_op",
    "phased_multidrive_rf_pulse_op",
    "initialize_mt_sat",
    "mt_sat_op",
    "multidrive_mt_sat_op",
    "rf_pulse",
    "rf_pulse_mt",
    "mt_sat",
]

from types import SimpleNamespace

import numpy as np
import scipy
import torch

# 1H Gyromagnetic Factor
gamma_bar = 42.577  # MHz / T
gamma = 2 * torch.pi * gamma_bar


def rf_pulse_op(
    fa: float,
    slice_prof: float | torch.Tensor = 1.0,
    B1: float = 1.0,
) -> tuple[tuple[torch.Tensor]]:
    """
    Build RF rotation matrix.

    Parameters
    ----------
    fa : float
        Nominal flip angle in ``rad``.
    slice_prof : float | torch.Tensor, optional
        Flip angle profile along slice. The default is ``1.0``.
    B1 : float, optional
        Flip angle scaling factor. The default is ``1.0``.

    Returns
    -------
    T : tuple[tuple[torch.Tensor]]
        RF rotation matrix elements.

    """
    # apply B1 effect
    fa = B1 * fa

    # apply slice profile
    fa = slice_prof * fa

    return _prep_rf(fa)


def phased_rf_pulse_op(
    fa: float,
    phi: float,
    slice_prof: float | torch.Tensor = 1.0,
    B1: float = 1.0,
    B1phase: float = 0.0,
) -> tuple[tuple[torch.Tensor]]:
    """
    Build RF rotation matrix along arbitrary axis.

    Parameters
    ----------
    fa : float
        Nominal flip angle in ``rad``.
    phi : float
        RF phase in ``rad``.
    slice_prof : float | torch.Tensor, optional
        Flip angle profile along slice. The default is ``1.0``.
    B1 : float, optional
        Flip angle scaling factor. The default is ``1.0``.
    B1phase : float, optional
        Transmit field phase in ``rad``. The default is ``0.0``.

    Returns
    -------
    T : tuple[tuple[torch.Tensor]]
        RF rotation matrix elements.

    """
    # apply B1 effect
    fa = B1 * fa
    phi = B1phase + phi

    # apply slice profile
    fa = slice_prof * fa

    return _prep_phased_rf(fa, phi)


def multidrive_rf_pulse_op(
    fa: torch.Tensor,
    slice_prof: float | torch.Tensor = 1.0,
    B1: float = 1.0,
) -> tuple[tuple[torch.Tensor]]:
    """
    Build RF rotation matrix for a multichannel RF pulse driven in phase.

    Every channel plays the same waveform with a real weight, so the array
    excites a voxel through the sum of what each channel puts there. Use
    :func:`phased_multidrive_rf_pulse_op` where the weights or the transmit
    sensitivities carry a phase: channels that differ in phase can cancel, and
    that only comes out of a complex sum.

    Parameters
    ----------
    fa : torch.Tensor
        Nominal flip angle in ``rad`` for each transmit channel.
        Expected shape is ``(nchannels,)``.
    slice_prof : float | torch.Tensor, optional
        Flip angle profile along slice. The default is ``1.0``.
    B1 : torch.Tensor, optional
        Flip angle scaling factor for each transmit channel.
        Expected shape is ``(nchannels,)``.
        The default is ``1.0``.

    Returns
    -------
    T : tuple[tuple[torch.Tensor]]
        RF rotation matrix elements.

    """
    # the array drives the voxel through the sum of its channels
    fa = (B1 * fa).sum(axis=-1)

    # apply slice profile
    fa = slice_prof * fa

    return _prep_rf(fa)


def phased_multidrive_rf_pulse_op(
    fa: torch.Tensor,
    phi: torch.Tensor,
    slice_prof: float | torch.Tensor = 1.0,
    B1: torch.Tensor = 1.0,
    B1phase: torch.Tensor = 0.0,
) -> tuple[tuple[tuple[torch.Tensor]], torch.Tensor]:
    """
    Build RF rotation matrix for a multichannel RF pulse along arbitrary axis.

    This is static parallel transmit: every channel plays the same waveform,
    weighted by a complex number that does not change during the pulse. The
    weights and the transmit sensitivities then combine into one complex field
    per voxel,

        B1(r) = sum_c  B1_c(r) * exp(1j * B1phase_c) * fa_c * exp(1j * phi_c)

    whose magnitude is the flip angle the voxel sees and whose argument is the
    axis it turns about. The sum is what lets channels cancel: two driven in
    antiphase leave the voxel untouched.

    Because the spatial factor is constant in time it comes outside the pulse
    integral, so this is exactly as accurate as the single-channel model with a
    transmit map -- no further approximation. A pulse whose channel weights
    vary during it does not factor that way and is not this operator.

    Parameters
    ----------
    fa : torch.Tensor
        Nominal flip angle in ``rad`` for each transmit channel.
        Expected shape is ``(nchannels,)``.
    phi : float
        RF phase in ``rad`` for each transmit channel.
        Expected shape is ``(nchannels,)``.
    slice_prof : float | torch.Tensor, optional
        Flip angle profile along slice. The default is 1.0.
    B1 : torch.Tensor, optional
        Flip angle scaling factor for each transmit channel.
        Expected shape is ``(nchannels,)``.
        The default is ``1.0``.
    B1phase : torch.Tensor, optional
        Transmit field phase in ``rad`` for each transmit channel.
        Expected shape is ``(nchannels,)``.
        The default is ``0.0``.

    Returns
    -------
    T : tuple[tuple[torch.Tensor]]
        RF rotation matrix elements.
    phi : torch.Tensor
        Nominal net RF phase for signal demodulation, which is the argument of
        the commanded drive alone: the receive reference is the phase that was
        asked for, not the one the transmit field added to it.

    """
    # the array drives the voxel through the complex sum of its channels
    commanded = (fa * torch.exp(1j * phi)).sum(axis=-1)
    field = (B1 * torch.exp(1j * B1phase) * fa * torch.exp(1j * phi)).sum(axis=-1)
    _fa = field.abs()
    _phi = field.angle()

    # apply slice profile
    _fa = slice_prof * _fa

    return _prep_phased_rf(_fa, _phi), commanded.angle()


def initialize_mt_sat(
    duration: float,
    b1rms: float,
    df: float = 0.0,
    slice_prof: float | torch.Tensor = 1.0,
    B1: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate RF energy deposition.

    This is the energy deposited by and RF pulse on a bound pool,
    i.e., a spin pool without transverse magnetization (e.g., T2 almost 0.0).

    Parameters
    ----------
    duration: float
        Pulse duration in ``[s]``.
    b1rms: float
        Pulse root-mean-squared B1 in ``[T]`` for each transmit channel.
        Expected shape is ``(nchannels,)``.
    df : float
        Frequency offset of the pulse in ``[Hz]``.
    slice_prof : float | torch.Tensor, optional
        Flip angle profile along slice. The default is ``1.0``.
    B1 : float, optional
        Flip angle scaling factor. The default is ``1.0``.

    Returns
    -------
    WT : torch.Tensor
        Energy yielded by the RF pulse.

    Notes
    -----
    When flip angle is constant throughout acquisition, user should use
    the output ``exp_WT``. When flip angle changes (e.g., MR Fingerprinting),
    user should provide the ``b1rms`` for a normalized pulse (i.e., when ``fa=1.0 [rad]``).
    Then, user rescale the output ``WT`` by the square of current flip angle, and use this
    to re-calculate ``exp_WT``. This can be achieved using the provided ``scale_mt_sat``
    function.

    Examples
    --------
    .. exec::
        :context: true

        import torch
        from torchsim.epg import initialize_mt_sat, mt_sat_op

    Constant flip angle case. We will use a pulse duration of 1ms
    and define B1rms so that the b1rms * tau is 32.7 uT**2 * ms.

    .. exec::
        :context: true

        duration = 1e-3 # 1ms pulse duration
        b1rms = 1e-6 * (32.7**0.5) / 1e-3 # B1 rms in [T]
        df = 0.0 # assume on-resonance pulse

    In this case, we can directly use the ``exp(WT)`` output.

    .. exec::
        :context: true

        WT = initialize_mt_sat(torch.as_tensor(duration), torch.as_tensor(b1rms), df, slice_prof=1.0, B1=1.0)
        S = mt_sat_op(WT)

    Variable flip angle case. We will use a pulse duration of 1ms
    and define B1rms so that the b1rms * tau is 54.3 uT**2 * ms when fa = 1 rad.

    .. exec::
        :context: true

        duration = 1e-3 # 1ms pulse duration
        b1rms = 1e-6 * (54.3**0.5) / 1e-3 # B1 rms in [T]
        df = 0.0 # assume on-resonance pulse

    In this case, we use the exponential argument only:

    .. exec::
        :context: true

        WT = initialize_mt_sat(torch.as_tensor(duration), torch.as_tensor(b1rms), df)

    Then, for each RF pulse of in the train, we rescale ``WT`` and recompute
    the exponential. Here, we do it explicitly:

    .. exec::
        :context: true

        fa = torch.linspace(5, 60.0, 1000)
        fa = torch.deg2rad(fa)
        for n in range(fa.shape[0]):
            # update saturation operator
            S = mt_sat_op(WT, fa[n], slice_prof=1.0, B1=1.0)

            # apply saturation here
            ...

    """
    # get parameters
    tau = duration * 1e3  # [ms]
    b1rms = b1rms * 1e6  # [uT]
    G = super_lorentzian_lineshape(df)

    # calculate WT
    W = torch.pi * (gamma * 1e-3) ** 2 * b1rms**2 * G
    WT = -W * tau

    return WT


def mt_sat_op(
    WT: torch.Tensor,
    fa: float = 1.0,
    slice_prof: float | torch.Tensor = 1.0,
    B1: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate RF saturation operator.

    This operator describes the effect of an RF pulse on a bound pool,
    i.e., a spin pool without transverse magnetization (e.g., T2 almost 0.0).

    Parameters
    ----------
    WT : torch.Tensor
        Energy yielded by the RF pulse.
    fa : float
        Nominal flip angle in ``rad``.
    slice_prof : float | torch.Tensor, optional
        Flip angle profile along slice. The default is ``1.0``.
    B1 : float, optional
        Flip angle scaling factor. The default is ``1.0``.

    Returns
    -------
    exp_WT : torch.Tensor
        RF saturation operator.
    """
    # apply B1 effect to fa
    fa = B1 * fa

    # apply slice profile
    fa = slice_prof * fa

    return torch.exp(fa**2 * WT)


def multidrive_mt_sat_op(
    WT: torch.Tensor,
    fa: torch.Tensor = 1.0,
    slice_prof: float | torch.Tensor = 1.0,
    B1: torch.Tensor = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build RF saturation matrix for a multuchannel pulse.

    This operator describes the effect of an RF pulse on a bound pool,
    i.e., a spin pool without transverse magnetization (e.g., T2 almost 0.0).

    Parameters
    ----------
    WT : torch.Tensor
        Energy yielded by the RF pulse.
    fa : torch.Tensor
        Nominal flip angle in ``rad`` for each transmit channel.
        Expected shape is ``(nchannels,)``.
    slice_prof : float | torch.Tensor, optional
        Flip angle profile along slice. The default is ``1.0``.
    B1 : float, optional
        Flip angle scaling factor. The default is ``1.0``.

    Returns
    -------
    exp_WT : torch.Tensor
        RF saturation operator.

    Notes
    -----
    Saturation goes as the square of the field, so the channels are summed
    before squaring rather than after. The sum is real, which assumes the array
    is driven in phase.
    """
    # the array saturates through the sum of its channels
    fa = (B1 * fa).sum(axis=-1)

    # apply slice profile
    fa = slice_prof * fa

    return torch.exp(fa**2 * WT)


def rf_pulse(
    states: SimpleNamespace,
    RF: tuple[tuple[torch.Tensor]],
) -> SimpleNamespace:
    """
    Apply RF rotation, mixing EPG states.

    Parameters
    ----------
    states : SimpleNamespace
        Input EPG states.
    RF : tuple[tuple[torch.Tensor]]
        RF rotation matrix elements.

    Returns
    -------
    SimpleNamespace
        Output EPG states.

    """
    FplusIn = states.Fplus
    FminusIn = states.Fminus
    ZIn = states.Z

    # prepare out
    FplusOut = FplusIn.clone()
    FminusOut = FminusIn.clone()
    ZOut = ZIn.clone()

    # apply
    FplusOut = RF[0][0] * FplusIn + RF[0][1] * FminusIn + RF[0][2] * ZIn
    FminusOut = RF[1][0] * FplusIn + RF[1][1] * FminusIn + RF[1][2] * ZIn
    ZOut = RF[2][0] * FplusIn + RF[2][1] * FminusIn + RF[2][2] * ZIn

    # prepare for output
    states.Fplus = FplusOut
    states.Fminus = FminusOut
    states.Z = ZOut

    return states


def rf_pulse_mt(
    states: SimpleNamespace,
    RF: tuple[tuple[torch.Tensor]],
) -> SimpleNamespace:
    """
    Apply RF rotation in presence of a bound pool, mixing free EPG states.

    Parameters
    ----------
    states : SimpleNamespace
        Input EPG states.
    RF : tuple[tuple[torch.Tensor]]
        RF rotation matrix elements.

    Returns
    -------
    SimpleNamespace
        Output EPG states.

    """
    FplusIn = states.Fplus
    FminusIn = states.Fminus
    ZIn = states.Z[..., :-1]

    # prepare out
    FplusOut = FplusIn.clone()
    FminusOut = FminusIn.clone()
    ZOut = ZIn.clone()

    # apply
    FplusOut = RF[0][0] * FplusIn + RF[0][1] * FminusIn + RF[0][2] * ZIn
    FminusOut = RF[1][0] * FplusIn + RF[1][1] * FminusIn + RF[1][2] * ZIn
    ZOut = RF[2][0] * FplusIn + RF[2][1] * FminusIn + RF[2][2] * ZIn

    # prepare for output
    states.Fplus = FplusOut
    states.Fminus = FminusOut
    states.Z[..., :-1] = ZOut

    return states


def mt_sat(
    states: SimpleNamespace,
    S: torch.Tensor,
) -> SimpleNamespace:
    """
    Apply RF saturation.

    Parameters
    ----------
    states : SimpleNamespace
        Input EPG states.
    S : torch.Tensor
        RF rsaturation operator.

    Returns
    -------
    SimpleNamespace
        Output EPG states.

    """
    Zbound = states.Z[
        ..., -1
    ]  # assume we have a single bound pool at last position in pool axis

    # prepare
    Zbound = S * Zbound.clone()

    # prepare for output
    states.Z = Zbound

    return states


# %% utils
def spinor_rf_pulse_op(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[tuple[torch.Tensor]]:
    """Build the RF rotation matrix from a pulse's Cayley-Klein pair.

    A pulse of any shape acts on a spin as one element of SU(2),

        U = [[a, b], [-conj(b), conj(a)]],   |a|^2 + |b|^2 = 1,

    and the EPG rotation is that element carried into the (F+, F-, Z) basis.
    Where :func:`phased_rf_pulse_op` names the two numbers an instantaneous
    pulse turns through, this names the four a real one leaves behind, which is
    what a shaped or slice-selective pulse needs: its rotation is about an
    arbitrary axis, and no flip-and-phase pair reaches those.

    Parameters
    ----------
    a : torch.Tensor
        Cayley-Klein ``a``, complex.
    b : torch.Tensor
        Cayley-Klein ``b``, complex.

    Returns
    -------
    T : tuple[tuple[torch.Tensor]]
        RF rotation matrix elements.

    Notes
    -----
    ``a = cos(fa / 2)`` and ``b = -1j * exp(-1j * phi) * sin(fa / 2)`` recovers
    :func:`phased_rf_pulse_op`.
    """
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)
    if not a.is_complex():
        a = a.to(torch.complex64)
    if not b.is_complex():
        b = b.to(torch.complex64)

    T00 = a.conj() ** 2
    T01 = -(b.conj() ** 2)
    T02 = -2.0 * (a * b).conj()
    T10 = -(b**2)
    T11 = a**2
    T12 = -2.0 * a * b
    T20 = a.conj() * b
    T21 = a * b.conj()
    T22 = a.abs() ** 2 - b.abs() ** 2

    T0 = [T00[..., None], T01[..., None], T02[..., None]]
    T1 = [T10[..., None], T11[..., None], T12[..., None]]
    T2 = [T20[..., None], T21[..., None], T22[..., None] + 0j]

    return [T0, T1, T2]


def _prep_rf(fa):
    # calculate operator
    T00 = torch.cos(fa / 2) ** 2
    T01 = torch.sin(fa / 2) ** 2
    T02 = -1j * torch.sin(fa)
    T10 = T01.conj()
    T11 = T00
    T12 = 1j * torch.sin(fa)
    T20 = -0.5 * 1j * torch.sin(fa)
    T21 = 0.5 * 1j * torch.sin(fa)
    T22 = torch.cos(fa)

    # build rows
    T0 = [T00[..., None], T01[..., None], T02[..., None]]
    T1 = [T10[..., None], T11[..., None], T12[..., None]]
    T2 = [T20[..., None], T21[..., None], T22[..., None]]

    # build matrix
    T = [T0, T1, T2]

    return T


def _prep_phased_rf(fa, phi):
    # calculate operator
    T00 = torch.cos(fa / 2) ** 2
    T01 = torch.exp(2 * 1j * phi) * (torch.sin(fa / 2)) ** 2
    T02 = -1j * torch.exp(1j * phi) * torch.sin(fa)
    T10 = T01.conj()
    T11 = T00
    T12 = 1j * torch.exp(-1j * phi) * torch.sin(fa)
    T20 = -0.5 * 1j * torch.exp(-1j * phi) * torch.sin(fa)
    T21 = 0.5 * 1j * torch.exp(1j * phi) * torch.sin(fa)
    T22 = torch.cos(fa)

    # build rows
    T0 = [T00[..., None], T01[..., None], T02[..., None]]
    T1 = [T10[..., None], T11[..., None], T12[..., None]]
    T2 = [T20[..., None], T21[..., None], T22[..., None]]

    # build matrix
    T = [T0, T1, T2]

    return T


def super_lorentzian_lineshape(
    df: float, T2star: float = 12e-6, fsample: tuple = (-30e3, 30e3)
) -> float:
    """
    Super Lorentzian lineshape.

    Parameters
    ----------
    df : float
        Frequency offset of the pulse in ``[Hz]``.
    T2star : float, optional
        T2 of semisolid compartment in ``s``.
        Defaults to ``12e-6`` (``12 us``).
    fsample : tuple, optional
        Frequency range at which function is to be evaluated in ``[Hz]``.
        Defaults to ``(-30e3, 30e3)``.

    Returns
    -------
    G: float
        Lineshape at the desired frequency ``df``.

    Example
    -------
    >>> from torchsim.epg._utils import super_lorentzian_lineshape
    >>> G = super_lorentzian_lineshape(0.5e3, 12e-6)

    Shaihan Malik (c), King's College London, April 2019
    Matteo Cencini: Python porting (December 2022)

    """
    # clone
    if isinstance(df, torch.Tensor):
        df = df.clone()
        df = df.cpu().numpy()
    else:
        df = np.asarray(df, dtype=np.float32)
    df = np.atleast_1d(df)

    # as suggested by Gloor, we can interpolate the lineshape across from
    # ± 1kHz
    nu = 100  # <-- number of points for theta integral

    # compute over a wider range of frequencies
    n = 128
    if fsample[0] > -30e3:
        fmin = -33e3
    else:
        fmin = 1.1 * fsample[0]

    if fsample[1] < 30e3:
        fmax = 33e3
    else:
        fmax = 1.1 * fsample[1]

    ff = np.linspace(fmin, fmax, n, dtype=np.float32)

    # np for Super Lorentzian, predefine
    u = np.linspace(0.0, 1.0, nu)
    du = np.diff(u)[0]

    # get integration grid
    ug, ffg = np.meshgrid(u, ff, indexing="ij")

    # prepare integrand
    g = np.sqrt(2 / np.pi) * T2star / np.abs(3 * ug**2 - 1)
    g = g * np.exp(-2 * (2 * np.pi * ffg * T2star / (3 * ug**2 - 1)) ** 2)

    # integrate over theta
    G = du * g.sum(axis=0)

    # interpolate zero frequency
    po = np.abs(ff) < 1e3  # points to interpolate
    pu = np.logical_not(po) * (
        np.abs(ff) < 2e3
    )  # points to use to estimate interpolator

    Gi = _spline(ff[pu], G[pu], ff[po])
    G[po] = Gi  # replace interpolated

    # calculate
    if np.isscalar(df):
        idx = np.argmin(abs(ff - df))
    else:
        idx = [np.argmin(abs(ff - f0)) for f0 in df]
        idx = np.asarray(idx)

    # get actual absorption (s)
    G = G[idx]

    return G


def _spline(x, y, xq):
    """
    Same as MATLAB cubic spline interpolation.
    """
    # interpolate
    cs = scipy.interpolate.InterpolatedUnivariateSpline(x, y)
    return cs(xq)
