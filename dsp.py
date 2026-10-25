"""
dsp.py -- all signal-processing code for PulseLab.

Contents
--------
1. Sample ECG generator (synthetic, so the app runs without internet)
2. Noise simulation (Gaussian, 50 Hz power line, baseline wander, combined)
3. Filter design + validation (FIR Hann, FIR Hamming, IIR Butterworth)
4. Zero-phase filtering
5. Metrics: MSE, RMSE, SNR (reference based)
6. Magnitude spectrum helper

SNR convention used everywhere in PulseLab
------------------------------------------
    x[n] = clean reference ECG,  y[n] = denoised output,  N = number of samples
    MSE  = (1/N) * sum (x[n] - y[n])^2
    RMSE = sqrt(MSE)
    SNR  = 10 * log10( sum x[n]^2 / sum (x[n] - y[n])^2 )   [dB]

i.e. signal power = power of the clean reference, "noise" = everything that
differs from the reference after filtering (residual noise + distortion).
Optionally the mean (DC offset) is removed from x and y first, because a
0.5 Hz high-pass can never preserve a DC offset.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, firwin, freqz, sosfiltfilt, sosfreqz

BASELINE_HZ = 0.3
POWERLINE_HZ = 50.0

NOISE_KINDS = [
    "Gaussian white noise",
    "50 Hz power-line interference",
    "Baseline wander",
    "Combined noise",
]

# Approximate transition widths of windowed-sinc FIR filters: df = K * fs / N
_FIR_K = {"hann": 3.1, "hamming": 3.3}
_FIR_STOPBAND_DB = {"hann": 44, "hamming": 53}


# --------------------------------------------------------------------------
# 1. Sample ECG
# --------------------------------------------------------------------------
def generate_sample_ecg(fs: float = 360.0, seconds: float = 10.0,
                        bpm: float = 72.0, seed: int = 7) -> np.ndarray:
    """Synthetic ECG-like signal (P, Q, R, S, T waves as Gaussians), in mV.

    It is NOT real patient data -- it is a clean, repeatable demo signal.
    Beat-to-beat interval varies by ~3 % so RR analysis has something to show.
    """
    rng = np.random.default_rng(seed)
    n = int(round(seconds * fs))
    t = np.arange(n) / fs
    x = np.zeros(n)
    # (centre offset from R peak [s], amplitude [mV], width sigma [s])
    waves = [(-0.20, 0.15, 0.025),   # P
             (-0.04, -0.12, 0.010),  # Q
             (0.00, 1.00, 0.011),    # R
             (0.035, -0.25, 0.012),  # S
             (0.26, 0.30, 0.045)]    # T
    rr_mean = 60.0 / bpm
    beat = 0.45
    while beat < seconds + 0.5:
        for off, amp, sig in waves:
            x += amp * np.exp(-0.5 * ((t - (beat + off)) / sig) ** 2)
        beat += rr_mean * (1.0 + 0.03 * rng.standard_normal())
    return x


# --------------------------------------------------------------------------
# 2. Noise simulation
# --------------------------------------------------------------------------
def generate_noise(clean: np.ndarray, fs: float, kind: str,
                   intensity_pct: float, seed: int = 42) -> np.ndarray:
    """Return a noise vector (same length as `clean`). `clean` is never modified.

    Amplitudes scale with the ECG's own standard deviation R so that the slider
    means the same thing for any dataset. At 100 % intensity:
        baseline wander amplitude  = 1.0 * R   (0.3 Hz sine)
        power-line amplitude       = 0.5 * R   (50 Hz sine)
        Gaussian noise std         = 0.5 * R
    """
    if kind not in NOISE_KINDS:
        raise ValueError(f"Unknown noise type: {kind}")
    clean = np.asarray(clean, dtype=float)
    n = len(clean)
    t = np.arange(n) / fs
    k = float(np.clip(intensity_pct, 0, 100)) / 100.0
    ref = float(np.std(clean)) or 1.0
    rng = np.random.default_rng(int(seed))

    need_powerline = kind in ("50 Hz power-line interference", "Combined noise")
    if need_powerline and POWERLINE_HZ >= fs / 2:
        raise ValueError(
            f"50 Hz power-line noise needs a sampling frequency above 100 Hz "
            f"(current: {fs:g} Hz)."
        )

    baseline = k * 1.0 * ref * np.sin(2 * np.pi * BASELINE_HZ * t)
    power = k * 0.5 * ref * np.sin(2 * np.pi * POWERLINE_HZ * t)
    gauss = k * 0.5 * ref * rng.standard_normal(n)

    if kind == "Gaussian white noise":
        return gauss
    if kind == "50 Hz power-line interference":
        return power
    if kind == "Baseline wander":
        return baseline
    return baseline + power + gauss


# --------------------------------------------------------------------------
# 3. Filter design + validation
# --------------------------------------------------------------------------
def effective_fir_order(order: int) -> int:
    """FIR order is forced even so the filter has an odd number of taps
    (symmetric, exactly linear phase)."""
    order = int(order)
    return order + (order % 2)


def fir_numtaps(order: int) -> int:
    return effective_fir_order(order) + 1


def validate_params(fs, low, high, fir_order, iir_order, n_samples) -> list[str]:
    """Return a list of human-readable problems (empty list = all good)."""
    errors: list[str] = []
    try:
        fs = float(fs)
    except (TypeError, ValueError):
        return ["Sampling frequency must be a number."]
    if not np.isfinite(fs) or fs <= 0:
        return ["Sampling frequency must be greater than 0 Hz."]
    nyq = fs / 2.0
    if low <= 0:
        errors.append("Lower cutoff must be greater than 0 Hz.")
    if high <= low:
        errors.append("Upper cutoff must be larger than the lower cutoff.")
    if high >= nyq:
        errors.append(
            f"Upper cutoff ({high:g} Hz) must be below the Nyquist frequency "
            f"({nyq:g} Hz = fs/2). Lower it or raise the sampling frequency."
        )
    if fir_order < 10 or fir_order > 2000:
        errors.append("FIR order must be between 10 and 2000.")
    if iir_order < 1 or iir_order > 10:
        errors.append("Butterworth order must be between 1 and 10.")
    if not errors:
        needed = 3 * fir_numtaps(fir_order) + 1
        if n_samples < needed:
            errors.append(
                f"Signal too short for a {fir_numtaps(fir_order)}-tap FIR filter: "
                f"{n_samples} samples available, at least {needed} needed "
                f"(zero-phase filtering pads the edges). Lower the FIR order or "
                f"use a longer segment."
            )
    return errors


def design_fir(window: str, fs: float, low: float, high: float, order: int):
    """Band-pass FIR via the window method. Same specs for every window."""
    return firwin(fir_numtaps(order), [low, high], pass_zero=False,
                  window=window, fs=fs)


def design_butter(fs: float, low: float, high: float, order: int):
    """Butterworth band-pass as second-order sections (numerically stable).
    Note: a band-pass of 'order N' has 2N poles (an 2N-th order filter)."""
    return butter(int(order), [low, high], btype="bandpass", fs=fs, output="sos")


def filter_specs(fs, low, high, fir_order, iir_order) -> dict:
    """Human-readable parameter summary for each filter (used by the UI/export)."""
    eff = effective_fir_order(fir_order)
    taps = eff + 1
    specs = {}
    for key, label in (("hann", "FIR Hann"), ("hamming", "FIR Hamming")):
        specs[key] = dict(
            name=label, kind="FIR (window method)", window=key.capitalize(),
            order=eff, taps=taps, low=low, high=high, fs=fs,
            transition_hz=_FIR_K[key] * fs / taps,
            stopband_db=_FIR_STOPBAND_DB[key],
        )
    specs["butter"] = dict(
        name="IIR Butterworth", kind="IIR (Butterworth)", window="-",
        order=int(iir_order), taps=None, low=low, high=high, fs=fs,
        transition_hz=None, stopband_db=None,
    )
    return specs


# --------------------------------------------------------------------------
# 4. Filtering
# --------------------------------------------------------------------------
def apply_filters(noisy, fs, low, high, fir_order, iir_order) -> dict:
    """Apply all three filters to the SAME input. Returns {key: signal}.

    Zero-phase (forward-backward) filtering is used so the outputs line up
    with the reference in time. This is for offline analysis; it is not a
    real-time causal implementation, and it squares the magnitude response.
    """
    x = np.asarray(noisy, dtype=float)
    errs = validate_params(fs, low, high, fir_order, iir_order, len(x))
    if errs:
        raise ValueError(" ".join(errs))
    out = {}
    for key in ("hann", "hamming"):
        b = design_fir(key, fs, low, high, fir_order)
        out[key] = filtfilt(b, [1.0], x)
    sos = design_butter(fs, low, high, iir_order)
    out["butter"] = sosfiltfilt(sos, x)
    return out


def frequency_responses(fs, low, high, fir_order, iir_order, n=4096) -> dict:
    """Single-pass magnitude responses in dB (what you design, before filtfilt)."""
    res = {}
    for key in ("hann", "hamming"):
        w, h = freqz(design_fir(key, fs, low, high, fir_order), worN=n, fs=fs)
        res[key] = (w, 20 * np.log10(np.maximum(np.abs(h), 1e-10)))
    w, h = sosfreqz(design_butter(fs, low, high, iir_order), worN=n, fs=fs)
    res["butter"] = (w, 20 * np.log10(np.maximum(np.abs(h), 1e-10)))
    return res


# --------------------------------------------------------------------------
# 5. Metrics
# --------------------------------------------------------------------------
def _align(ref, est):
    ref = np.asarray(ref, dtype=float)
    est = np.asarray(est, dtype=float)
    n = min(len(ref), len(est))
    if n == 0:
        raise ValueError("Empty signal.")
    return ref[:n], est[:n]


def mse(ref, est, remove_dc: bool = False) -> float:
    r, e = _align(ref, est)
    if remove_dc:
        r, e = r - r.mean(), e - e.mean()
    return float(np.mean((r - e) ** 2))


def rmse(ref, est, remove_dc: bool = False) -> float:
    return float(np.sqrt(mse(ref, est, remove_dc)))


def snr_db(ref, est, remove_dc: bool = False) -> float:
    r, e = _align(ref, est)
    if remove_dc:
        r, e = r - r.mean(), e - e.mean()
    err_p = float(np.sum((r - e) ** 2))
    sig_p = float(np.sum(r ** 2))
    if err_p == 0:
        return float("inf")
    if sig_p == 0:
        return float("-inf")
    return float(10 * np.log10(sig_p / err_p))


def compute_metrics(ref, outputs: dict, remove_dc: bool = True) -> dict:
    """{filter_key: {"snr_db", "mse", "rmse"}} computed from the actual signals."""
    return {
        k: dict(snr_db=snr_db(ref, y, remove_dc), mse=mse(ref, y, remove_dc),
                rmse=rmse(ref, y, remove_dc))
        for k, y in outputs.items()
    }


# --------------------------------------------------------------------------
# 6. Spectrum
# --------------------------------------------------------------------------
def magnitude_spectrum(x, fs):
    """Single-sided amplitude spectrum (not a PSD)."""
    from scipy.fft import rfft, rfftfreq
    x = np.asarray(x, dtype=float)
    spec = np.abs(rfft(x)) / len(x)
    if len(spec) > 2:
        spec[1:-1] *= 2
    return rfftfreq(len(x), d=1.0 / fs), spec
