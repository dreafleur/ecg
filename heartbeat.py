"""
heartbeat.py -- simple, explainable R-peak detection and RR-interval analysis.

EDUCATIONAL ONLY. Irregular RR intervals alone cannot establish an arrhythmia,
and nothing here is clinically validated.

Pipeline (Pan-Tompkins-style, simplified)
-----------------------------------------
1. Band-pass 5-15 Hz      -> keeps QRS energy, drops P/T waves and drift
2. Derivative, then square -> QRS has the steepest slopes
3. Moving average (120 ms) -> smooth "energy envelope", one hump per beat
4. Peak picking on envelope with a 250 ms refractory distance
5. Keep humps that are >= 30 % of a typical strong hump (adaptive threshold)
6. Refine each peak to the true local maximum of the ECG (+/- 75 ms)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

REGULAR = "Appears Regular"
IRREGULAR = "Possible Irregularity Detected"
INSUFFICIENT = "Insufficient Signal Quality"


@dataclass
class RhythmConfig:
    min_rr_ms: float = 300.0        # shorter RR = physiologically implausible
    max_rr_ms: float = 2000.0       # longer RR = probably a missed beat
    cv_threshold: float = 0.12      # CV (SD/mean) above this -> irregular
    jump_threshold: float = 0.20    # |dRR|/RR above this counts as a "jump"
    jump_fraction: float = 0.20     # more than this share of jumps -> irregular
    min_beats: int = 6              # fewer detected beats -> insufficient
    max_bad_fraction: float = 0.20  # too many implausible RRs -> insufficient
    min_duration_s: float = 5.0
    min_beat_corr: float = 0.60     # beats must resemble each other (template match)


def beat_template_correlation(x, peaks, fs) -> float | None:
    """Median Pearson correlation between each beat (+/-200 ms around the R
    peak) and the median beat. Real ECG beats look alike (~0.9); detections
    on pure noise do not (~0.1-0.3). Used as a simple signal-quality index."""
    half = int(0.2 * fs)
    segs = [x[p - half: p + half + 1] for p in peaks if p - half >= 0 and p + half + 1 <= len(x)]
    if len(segs) < 3:
        return None
    segs = np.array(segs)
    segs = segs - segs.mean(axis=1, keepdims=True)
    template = np.median(segs, axis=0)
    if np.std(template) == 0:
        return 0.0
    cors = [np.corrcoef(s, template)[0, 1] for s in segs if np.std(s) > 0]
    return float(np.median(cors)) if cors else 0.0


@dataclass
class HeartbeatResult:
    verdict: str = INSUFFICIENT
    reasons: list = field(default_factory=list)
    peaks: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    peak_times_s: np.ndarray = field(default_factory=lambda: np.array([]))
    rr_ms: np.ndarray = field(default_factory=lambda: np.array([]))
    rr_times_s: np.ndarray = field(default_factory=lambda: np.array([]))
    rr_valid: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    hr_bpm: float | None = None
    mean_rr_ms: float | None = None
    sd_rr_ms: float | None = None
    cv: float | None = None
    rmssd_ms: float | None = None
    jump_frac: float | None = None
    beat_corr: float | None = None


def detect_rpeaks(x, fs) -> np.ndarray:
    """Return sample indices of candidate R peaks (empty array on failure)."""
    x = np.asarray(x, dtype=float)
    empty = np.array([], dtype=int)
    if len(x) < int(2 * fs) or not np.all(np.isfinite(x)):
        return empty
    nyq = fs / 2.0
    hi = min(15.0, 0.9 * nyq)
    lo = 5.0 if hi > 8 else 0.5 * hi
    try:
        sos = butter(2, [lo, hi], btype="bandpass", fs=fs, output="sos")
        bp = sosfiltfilt(sos, x)
    except Exception:
        return empty

    env = np.gradient(bp) ** 2
    win = max(1, int(0.12 * fs))
    env = np.convolve(env, np.ones(win) / win, mode="same")
    if env.max() <= 0:
        return empty

    cand, _ = find_peaks(env, distance=max(1, int(0.25 * fs)))
    if len(cand) == 0:
        return empty
    heights = env[cand]
    top = np.sort(heights)[::-1][: max(3, len(heights) // 3)]
    cand = cand[heights >= 0.30 * np.median(top)]
    if len(cand) == 0:
        return empty

    # polarity: is the QRS mostly an upward or a downward deflection?
    half = int(0.075 * fs)
    pos = neg = 0.0
    for c in cand:
        seg = bp[max(0, c - half): c + half + 1]
        pos += seg.max()
        neg += -seg.min()
    pol = 1.0 if pos >= neg else -1.0

    refined = []
    for c in cand:
        a, b = max(0, c - half), min(len(x), c + half + 1)
        refined.append(a + int(np.argmax(pol * x[a:b])))
    refined = np.unique(refined)

    # drop duplicates closer than 250 ms, keeping the stronger one
    min_d = int(0.25 * fs)
    kept: list[int] = []
    for p in refined:
        if kept and p - kept[-1] < min_d:
            if pol * x[p] > pol * x[kept[-1]]:
                kept[-1] = p
        else:
            kept.append(int(p))
    return np.array(kept, dtype=int)


def analyze_rhythm(x, fs, cfg: RhythmConfig | None = None) -> HeartbeatResult:
    """Detect beats, compute RR statistics and return a threshold-based verdict."""
    cfg = cfg or RhythmConfig()
    res = HeartbeatResult()
    x = np.asarray(x, dtype=float)

    if len(x) / fs < cfg.min_duration_s:
        res.reasons.append(
            f"Signal is {len(x) / fs:.1f} s long; at least {cfg.min_duration_s:g} s is needed.")
        return res

    peaks = detect_rpeaks(x, fs)
    res.peaks = peaks
    res.peak_times_s = peaks / fs
    if len(peaks) < cfg.min_beats:
        res.reasons.append(
            f"Only {len(peaks)} beat(s) detected; at least {cfg.min_beats} are needed.")
        return res

    res.beat_corr = beat_template_correlation(x, peaks, fs)
    if res.beat_corr is not None and res.beat_corr < cfg.min_beat_corr:
        res.reasons.append(
            f"Detected beats do not resemble each other (template correlation "
            f"{res.beat_corr:.2f} < {cfg.min_beat_corr:.2f}); the signal is too noisy "
            f"for reliable peak detection.")
        return res

    rr = np.diff(peaks) / fs * 1000.0
    valid = (rr >= cfg.min_rr_ms) & (rr <= cfg.max_rr_ms)
    res.rr_ms = rr
    res.rr_times_s = peaks[1:] / fs
    res.rr_valid = valid

    bad_frac = 1.0 - valid.mean()
    if bad_frac > cfg.max_bad_fraction:
        res.reasons.append(
            f"{bad_frac * 100:.0f}% of RR intervals are outside "
            f"{cfg.min_rr_ms:g}-{cfg.max_rr_ms:g} ms (likely false or missed detections).")
        return res
    good = rr[valid]
    if len(good) < 3:
        res.reasons.append("Fewer than 3 plausible RR intervals.")
        return res

    mean_rr = float(good.mean())
    sd = float(good.std(ddof=1))
    res.mean_rr_ms = mean_rr
    res.sd_rr_ms = sd
    res.cv = sd / mean_rr
    res.hr_bpm = 60000.0 / mean_rr
    d = np.diff(rr)
    pair_ok = valid[:-1] & valid[1:]
    res.rmssd_ms = float(np.sqrt(np.mean(d[pair_ok] ** 2))) if pair_ok.any() else None
    if pair_ok.any():
        res.jump_frac = float(np.mean(np.abs(d[pair_ok]) / rr[:-1][pair_ok] > cfg.jump_threshold))
    else:
        res.jump_frac = 0.0

    if not (30 <= res.hr_bpm <= 220):
        res.reasons.append(
            f"Estimated rate of {res.hr_bpm:.0f} BPM is outside 30-220 BPM; "
            f"peak detection is probably unreliable.")
        return res

    flags = []
    if res.cv > cfg.cv_threshold:
        flags.append(f"RR variability (CV {res.cv * 100:.1f}%) exceeds {cfg.cv_threshold * 100:.0f}%.")
    if res.jump_frac > cfg.jump_fraction:
        flags.append(
            f"{res.jump_frac * 100:.0f}% of beat-to-beat changes exceed "
            f"{cfg.jump_threshold * 100:.0f}% (limit {cfg.jump_fraction * 100:.0f}%).")
    if flags:
        res.verdict = IRREGULAR
        res.reasons = flags
    else:
        res.verdict = REGULAR
        res.reasons = [f"CV {res.cv * 100:.1f}% and beat-to-beat changes are within the thresholds."]
    return res
