"""
app.py -- PulseLab: ECG Signal Denoising & Analysis (Streamlit interface).

Run with:   python -m streamlit run app.py

This file only handles the interface, state and graphs.
  dsp.py        -> filters, noise, SNR / MSE / RMSE
  heartbeat.py  -> R-peak detection and RR-interval regularity
"""
from __future__ import annotations

import hashlib
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import dsp
import heartbeat as hb

st.set_page_config(page_title="PulseLab — ECG Signal Denoising & Analysis",
                   page_icon="🫀", layout="wide", initial_sidebar_state="expanded")

S = st.session_state

# --------------------------------------------------------------------------
# Theme
# --------------------------------------------------------------------------
C = dict(bg="#100B1F", card="#1B1430", panel="#241B3D", violet="#A78BFA",
         cyan="#38D9F8", mint="#4DFFB5", hann="#38D9F8", hamming="#B794F6",
         butter="#FF6FAE", noise="#FFB454", text="#F3EEFF", muted="#B7A9D6",
         grid="#3A2E5C", border="#33275A", plot="#130D24")

FILTERS = [
    ("hann", "FIR — Hann Window", "FIR Hann", C["hann"]),
    ("hamming", "FIR — Hamming Window", "FIR Hamming", C["hamming"]),
    ("butter", "IIR — Butterworth", "IIR Butterworth", C["butter"]),
]
FCOLOR = {k: c for k, _, _, c in FILTERS}
FSHORT = {k: s for k, _, s, _ in FILTERS}
FTITLE = {k: t for k, t, _, _ in FILTERS}

PAGES = [
    ("Dashboard", ":material/dashboard:"),
    ("Import ECG", ":material/upload_file:"),
    ("Noise Simulation", ":material/graphic_eq:"),
    ("Filter Comparison", ":material/tune:"),
    ("Heartbeat Analysis", ":material/monitor_heart:"),
    ("About the Project", ":material/info:"),
]

CSS = f"""
<style>
.stApp {{ background: {C['bg']}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1600px; }}
h1,h2,h3,h4 {{ color: {C['text']}; letter-spacing: -0.01em; }}

/* sidebar */
section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, {C['card']} 0%, {C['bg']} 100%);
  border-right: 1px solid {C['border']};
}}
section[data-testid="stSidebar"] button[kind="secondary"] {{
  background: transparent; border: 1px solid transparent; color: {C['muted']};
  justify-content: flex-start; border-radius: 12px; padding: .55rem .8rem;
}}
section[data-testid="stSidebar"] button[kind="secondary"]:hover {{
  background: {C['panel']}; color: {C['text']}; border-color: {C['border']};
}}
section[data-testid="stSidebar"] button[kind="primary"] {{
  background: linear-gradient(90deg, {C['violet']}, #8B6CF0); color: #14092B;
  border: none; font-weight: 700; justify-content: flex-start; border-radius: 12px;
  padding: .55rem .8rem; box-shadow: 0 0 18px rgba(167,139,250,.35);
}}

/* bordered containers = cards */
div[data-testid="stVerticalBlockBorderWrapper"] {{
  background: {C['card']}; border-radius: 16px; border-color: {C['border']};
}}
.st-key-card_hann {{ border-top: 3px solid {C['hann']} !important; }}
.st-key-card_hamming {{ border-top: 3px solid {C['hamming']} !important; }}
.st-key-card_butter {{ border-top: 3px solid {C['butter']} !important; }}

/* primary buttons in main area */
.main button[kind="primary"] {{
  background: linear-gradient(90deg, {C['violet']}, {C['cyan']}); color: #14092B;
  border: none; font-weight: 700; border-radius: 12px;
  box-shadow: 0 0 20px rgba(167,139,250,.35);
}}
.main button[kind="secondary"] {{
  background: {C['panel']}; border: 1px solid {C['border']}; color: {C['text']}; border-radius: 12px;
}}
.main button[kind="secondary"]:hover {{ border-color: {C['violet']}; color: {C['violet']}; }}

[data-testid="stFileUploaderDropzone"] {{
  background: {C['card']}; border: 1.5px dashed {C['violet']}; border-radius: 14px;
}}
div[data-testid="stExpander"] {{ background: {C['card']}; border: 1px solid {C['border']}; border-radius: 14px; }}
button[data-baseweb="tab"] {{ color: {C['muted']}; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color: {C['violet']}; }}

/* custom html components */
.pl-title {{ font-size: 1.9rem; font-weight: 800; color: {C['text']}; line-height: 1.15; }}
.pl-sub {{ color: {C['muted']}; font-size: 1rem; margin-top: .2rem; }}
.pl-pill {{ display:inline-block; padding: 6px 14px; border-radius: 999px; font-size:.85rem;
  background: {C['panel']}; border:1px solid {C['border']}; color: {C['text']}; }}
.pl-dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:8px; }}
.pl-kpi {{ background: {C['card']}; border:1px solid {C['border']}; border-radius:16px; padding:14px 16px; height:100%; }}
.pl-kpi .l {{ color:{C['muted']}; font-size:.74rem; text-transform:uppercase; letter-spacing:.07em; }}
.pl-kpi .v {{ color:{C['text']}; font-size:1.7rem; font-weight:750; line-height:1.25; }}
.pl-kpi .s {{ color:{C['muted']}; font-size:.8rem; }}
.pl-chip {{ display:inline-block; padding:3px 10px; margin:2px 5px 2px 0; border-radius:999px;
  background:{C['panel']}; border:1px solid {C['border']}; font-size:.78rem; color:{C['text']}; }}
.pl-card-title {{ font-size:1.05rem; font-weight:700; }}
.pl-note {{ background:{C['panel']}; border-left: 3px solid {C['violet']}; border-radius:8px;
  padding:10px 14px; color:{C['muted']}; font-size:.88rem; }}
.pl-warn {{ background:#2A1F1A; border-left: 3px solid {C['noise']}; border-radius:8px;
  padding:10px 14px; color:#FFE2B8; font-size:.9rem; }}
table.pl-table {{ width:100%; border-collapse:collapse; background:{C['card']}; border-radius:14px; overflow:hidden; }}
table.pl-table th {{ background:{C['panel']}; color:{C['muted']}; text-align:left; font-size:.78rem;
  text-transform:uppercase; letter-spacing:.06em; padding:11px 14px; }}
table.pl-table td {{ padding:11px 14px; border-top:1px solid {C['border']}; color:{C['text']}; font-size:.95rem; }}
.pl-best {{ color:{C['mint']}; font-weight:700; }}
.pl-flow {{ display:flex; flex-wrap:wrap; align-items:center; gap:8px; }}
.pl-box {{ background:{C['panel']}; border:1px solid {C['violet']}; border-radius:12px; padding:9px 14px;
  font-weight:600; font-size:.9rem; }}
.pl-arrow {{ color:{C['violet']}; font-size:1.3rem; }}
.pl-verdict {{ border-radius:16px; padding:18px 22px; border:1px solid; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
DEFAULTS = dict(
    page="Dashboard", sig=None, has_ref=False, fs=360.0, fname=None, results=None,
    run_errors=None, view_win=None,
    noise_kind="Combined noise", noise_int=30, noise_seed=42,
    low_cut=0.5, high_cut=40.0, fir_order=100, iir_order=4, remove_dc=True,
    hb_source="hann", hb_cv=12, hb_jump=20, hb_jump_frac=20, hb_min_beats=6,
    hb_min_rr=300, hb_max_rr=2000, hb_bad_frac=20, hb_corr=60,
    show_clean=True, show_noisy=True, show_hann=True, show_hamming=True, show_butter=True,
    import_mode="Clean reference ECG", seg_seconds=10.0,
)
for k, v in DEFAULTS.items():
    S.setdefault(k, v)
# Streamlit forgets widget values for widgets that are not drawn on the current
# page; re-assigning keeps them alive while the user navigates.
for k in DEFAULTS:
    if k in S and k not in ("view_win",):
        S[k] = S[k]

PLOT_CFG = dict(scrollZoom=True, displaylogo=False,
                modeBarButtonsToRemove=["lasso2d", "select2d"])


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def duration() -> float:
    return len(S.sig) / S.fs if S.sig is not None else 0.0


def kpi(label, value, sub="", color=None):
    style = f' style="color:{color}"' if color else ""
    return (f'<div class="pl-kpi"><div class="l">{label}</div>'
            f'<div class="v"{style}>{value}</div><div class="s">{sub}</div></div>')


def chips(items):
    return "".join(f'<span class="pl-chip">{i}</span>' for i in items)


def fmt_mse(v):
    if not np.isfinite(v):
        return "—"
    return f"{v:.2e}" if (v < 1e-4 and v != 0) else f"{v:.5f}"


def fmt_snr(v):
    return f"{v:.2f}" if np.isfinite(v) else ("∞" if v > 0 else "−∞")


def base_fig(height=300, xrange=None, yrange=None, ytitle="Amplitude (mV)"):
    fig = go.Figure()
    fig.update_layout(
        height=height, paper_bgcolor=C["card"], plot_bgcolor=C["plot"],
        font=dict(color=C["text"], size=13), margin=dict(l=60, r=18, t=42, b=48),
        legend=dict(orientation="h", y=1.14, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified", dragmode="zoom",
        xaxis=dict(title="Time (s)", gridcolor=C["grid"], zerolinecolor=C["grid"],
                   range=xrange, linecolor=C["grid"]),
        yaxis=dict(title=ytitle, gridcolor=C["grid"], zerolinecolor=C["grid"],
                   range=yrange, linecolor=C["grid"]),
    )
    return fig


def add_line(fig, x, y, name, color, width=1.6, opacity=1.0, dash=None):
    cls = go.Scattergl if len(x) > 20000 else go.Scatter
    fig.add_trace(cls(x=x, y=y, name=name, mode="lines", opacity=opacity,
                      line=dict(color=color, width=width, dash=dash)))


def show(fig, key=None):
    st.plotly_chart(fig, width="stretch", config=PLOT_CFG, key=key)


def get_params():
    return dict(fs=float(S.fs), low=float(S.low_cut), high=float(S.high_cut),
                fir_order=int(S.fir_order), iir_order=int(S.iir_order))


def compute_noisy():
    """Return (noisy, noise, error). The clean reference S.sig is never modified."""
    if S.sig is None:
        return None, None, None
    if not S.has_ref:
        return S.sig.copy(), None, None
    try:
        noise = dsp.generate_noise(S.sig, S.fs, S.noise_kind, S.noise_int, S.noise_seed)
        return S.sig + noise, noise, None
    except ValueError as e:
        return S.sig.copy(), None, str(e)


def signature():
    """Fingerprint of everything that influences the filter results."""
    h = hashlib.md5()
    h.update(np.ascontiguousarray(S.sig).tobytes())
    h.update(repr((S.has_ref, S.fs, S.low_cut, S.high_cut, S.fir_order, S.iir_order,
                   S.remove_dc, S.noise_kind, S.noise_int, S.noise_seed)).encode())
    return h.hexdigest()


def rhythm_cfg():
    return hb.RhythmConfig(
        min_rr_ms=S.hb_min_rr, max_rr_ms=S.hb_max_rr, cv_threshold=S.hb_cv / 100,
        jump_threshold=S.hb_jump / 100, jump_fraction=S.hb_jump_frac / 100,
        min_beats=int(S.hb_min_beats), max_bad_fraction=S.hb_bad_frac / 100,
        min_beat_corr=S.hb_corr / 100)


def goto(page):
    S.page = page


# --------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------
def _reset_loaded(sig, has_ref, fs, name):
    S.sig, S.has_ref, S.fs, S.fname = sig, has_ref, float(fs), name
    S.results, S.run_errors, S.view_win = None, None, None


def load_sample():
    _reset_loaded(dsp.generate_sample_ecg(360.0, 10.0), True, 360.0,
                  "sample_ecg_360Hz (synthetic)")


def reset_noise():
    S.noise_kind, S.noise_int, S.noise_seed = "Combined noise", 30, 42


def new_noise_seed():
    S.noise_seed = int(np.random.randint(1, 1_000_000))


def run_filters():
    S.run_errors = None
    if S.sig is None:
        S.run_errors = ["Load an ECG first (Import ECG)."]
        return
    noisy, _, err = compute_noisy()
    if err:
        S.run_errors = [err]
        return
    p = get_params()
    errs = dsp.validate_params(p["fs"], p["low"], p["high"], p["fir_order"],
                               p["iir_order"], len(noisy))
    if errs:
        S.run_errors = errs
        return
    try:
        outs = dsp.apply_filters(noisy, **p)
        metrics = dsp.compute_metrics(S.sig, outs, S.remove_dc) if S.has_ref else None
    except Exception as e:  # keep the app alive, show a helpful message
        S.run_errors = [f"Filtering failed: {e}"]
        return
    S.results = dict(outputs=outs, metrics=metrics, params=p, noisy=noisy,
                     sig=signature(), has_ref=S.has_ref, remove_dc=S.remove_dc,
                     noise=(S.noise_kind, S.noise_int, S.noise_seed) if S.has_ref else None)


def results_stale():
    return S.results is not None and S.results["sig"] != signature()


# --------------------------------------------------------------------------
# Sidebar + header
# --------------------------------------------------------------------------
LOGO = f"""
<div style="display:flex;align-items:center;gap:10px;margin:4px 0 18px 4px;">
<svg width="38" height="38" viewBox="0 0 38 38"><rect width="38" height="38" rx="11" fill="{C['panel']}" stroke="{C['violet']}"/>
<polyline points="5,21 12,21 15,10 20,29 24,16 27,21 33,21" fill="none" stroke="{C['mint']}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>
<div><div style="font-weight:800;font-size:1.25rem;color:{C['text']};line-height:1">PulseLab</div>
<div style="font-size:.72rem;color:{C['muted']};letter-spacing:.08em">DSP · ECG LAB</div></div></div>
"""

with st.sidebar:
    st.markdown(LOGO, unsafe_allow_html=True)
    for name, icon in PAGES:
        st.button(name, icon=icon, key=f"nav_{name}", width="stretch",
                  type="primary" if S.page == name else "secondary",
                  on_click=goto, args=(name,))
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="pl-note">Educational DSP demo — not a clinical or diagnostic tool.</div>',
                unsafe_allow_html=True)


def build_exports():
    """Return dict of export payloads, only from actually computed results."""
    R = S.results
    if R is None:
        return None
    p = R["params"]
    n = len(R["noisy"])
    cols = {"time_s": np.arange(n) / p["fs"]}
    if R["has_ref"]:
        cols["clean_reference"] = S.sig[:n]
    cols["noisy_input" if R["has_ref"] else "imported_input"] = R["noisy"]
    for k, _, short, _ in FILTERS:
        cols[k if k != "butter" else "iir_butterworth"] = R["outputs"][k]
    cols = {("fir_" + k if k in ("hann", "hamming") else k): v for k, v in cols.items()}
    signals_csv = pd.DataFrame(cols).to_csv(index=False)

    specs = dsp.filter_specs(p["fs"], p["low"], p["high"], p["fir_order"], p["iir_order"])
    metrics_csv = None
    if R["metrics"]:
        rows = []
        for k, _, short, _ in FILTERS:
            sp, m = specs[k], R["metrics"][k]
            rows.append({
                "filter": short, "type": sp["kind"], "window": sp["window"],
                "order": sp["order"], "num_taps": sp["taps"],
                "fs_hz": p["fs"], "low_cutoff_hz": p["low"], "high_cutoff_hz": p["high"],
                "noise_type": R["noise"][0], "noise_intensity_pct": R["noise"][1],
                "noise_seed": R["noise"][2], "dc_offset_removed": R["remove_dc"],
                "snr_db": m["snr_db"], "mse": m["mse"], "rmse": m["rmse"]})
        metrics_csv = pd.DataFrame(rows).to_csv(index=False)

    L = ["# PulseLab — Filter Comparison Report", "",
         f"- Signal: {S.fname}  ({n} samples, {p['fs']:g} Hz, {n / p['fs']:.2f} s)",
         f"- Band-pass: {p['low']:g}–{p['high']:g} Hz, zero-phase (forward-backward) filtering"]
    if R["has_ref"]:
        L.append(f"- Simulated noise: {R['noise'][0]}, {R['noise'][1]}% intensity, seed {R['noise'][2]}")
    L += ["", "## Filters", "",
          "| Filter | Type | Order | Taps | Band (Hz) |", "|---|---|---|---|---|"]
    for k, _, short, _ in FILTERS:
        sp = specs[k]
        L.append(f"| {short} | {sp['kind']} | {sp['order']} | {sp['taps'] or '-'} | {sp['low']:g}–{sp['high']:g} |")
    L += ["", "## Performance", ""]
    if R["metrics"]:
        L += ["SNR = 10·log10(Σx² / Σ(x−y)²), x = clean reference, y = filtered output. "
              f"MSE = mean((x−y)²), RMSE = √MSE. DC offset removed before metrics: {R['remove_dc']}.", "",
              "| Filter | SNR (dB) | MSE | RMSE |", "|---|---|---|---|"]
        for k, _, short, _ in FILTERS:
            m = R["metrics"][k]
            L.append(f"| {short} | {m['snr_db']:.3f} | {m['mse']:.6g} | {m['rmse']:.6g} |")
        best = max(R["metrics"], key=lambda k: R["metrics"][k]["snr_db"])
        L.append(f"\nHighest SNR / lowest MSE in this run: {FSHORT[best]}. "
                 "This result applies only to this signal, noise setting and parameters.")
    else:
        L.append("Reference required: the imported signal has no clean reference, so SNR/MSE/RMSE were not calculated.")
    L += ["", "## Heartbeat analysis (educational, not diagnostic)", "",
          "| Signal | R-peaks | HR (BPM) | Mean RR (ms) | RR SD (ms) | CV (%) | Result |", "|---|---|---|---|---|---|---|"]
    for k, _, short, _ in FILTERS:
        r = hb.analyze_rhythm(R["outputs"][k], p["fs"], rhythm_cfg())
        f = lambda v, d=1: "-" if v is None else f"{v:.{d}f}"
        L.append(f"| {short} | {len(r.peaks)} | {f(r.hr_bpm)} | {f(r.mean_rr_ms)} | {f(r.sd_rr_ms)} | "
                 f"{f(None if r.cv is None else r.cv * 100)} | {r.verdict} |")
    L += ["", "> Heartbeat regularity is estimated using signal-processing methods for educational purposes "
          "only. This tool cannot diagnose arrhythmias or replace professional medical evaluation."]
    return dict(signals=signals_csv, metrics=metrics_csv, report="\n".join(L))


def header():
    left, mid, right = st.columns([5, 2.6, 1.6], vertical_alignment="center")
    with left:
        st.markdown('<div class="pl-title">ECG Signal Denoising &amp; Analysis</div>'
                    '<div class="pl-sub">Explore digital filters, visualize noise reduction, '
                    'and evaluate ECG signal quality.</div>', unsafe_allow_html=True)
    with mid:
        if S.sig is None:
            st.markdown(f'<span class="pl-pill"><span class="pl-dot" style="background:{C["noise"]}"></span>'
                        f'No ECG loaded</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="pl-pill"><span class="pl-dot" style="background:{C["mint"]}"></span>'
                        f'ECG loaded · {S.fs:g} Hz · {duration():.1f} s</span>', unsafe_allow_html=True)
    with right:
        with st.popover("Export Results", icon=":material/download:", width="stretch"):
            ex = build_exports()
            if ex is None:
                st.caption("Run **Apply Filters & Compare** first — nothing is exported until real results exist.")
            else:
                st.download_button("Filtered signals (CSV)", ex["signals"], "pulselab_filtered_signals.csv",
                                   "text/csv", on_click="ignore", width="stretch")
                if ex["metrics"]:
                    st.download_button("Performance metrics (CSV)", ex["metrics"], "pulselab_metrics.csv",
                                       "text/csv", on_click="ignore", width="stretch")
                else:
                    st.caption("Metrics CSV unavailable: reference required.")
                st.download_button("Comparison report (.md)", ex["report"], "pulselab_report.md",
                                   "text/markdown", on_click="ignore", width="stretch")
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)


def need_signal() -> bool:
    if S.sig is None:
        st.markdown('<div class="pl-warn">No ECG loaded yet. Load the sample or import a CSV to continue.</div>',
                    unsafe_allow_html=True)
        c1, c2, _ = st.columns([1.2, 1.2, 4])
        c1.button("Load Sample ECG", on_click=load_sample, type="primary", key=f"ns_sample_{S.page}")
        c2.button("Go to Import ECG", on_click=goto, args=("Import ECG",), key=f"ns_imp_{S.page}")
        return True
    return False


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
FLOW = ["Clean ECG", "Noise Simulation", "Noisy ECG", "Three Denoising Filters",
        "Performance Comparison", "Heartbeat Analysis"]


def flow_html():
    parts = []
    for i, b in enumerate(FLOW):
        parts.append(f'<div class="pl-box">{b}</div>')
        if i < len(FLOW) - 1:
            parts.append('<div class="pl-arrow">→</div>')
    return f'<div class="pl-flow">{"".join(parts)}</div>'


def page_dashboard():
    c = st.columns(4)
    if S.sig is None:
        c[0].markdown(kpi("Signal", "—", "none loaded", C["noise"]), unsafe_allow_html=True)
        c[1].markdown(kpi("Sampling", "—", "Hz"), unsafe_allow_html=True)
        c[2].markdown(kpi("Duration", "—", "seconds"), unsafe_allow_html=True)
        c[3].markdown(kpi("Reference", "—", ""), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("#### Get started")
            st.markdown("1. **Load Sample ECG** (or import your own CSV)\n"
                        "2. Add simulated noise in **Noise Simulation**\n"
                        "3. Press **Apply Filters & Compare**\n"
                        "4. Check **Heartbeat Analysis** on the denoised signal")
            st.button("Load Sample ECG", on_click=load_sample, type="primary", key="dash_sample")
    else:
        c[0].markdown(kpi("Signal", "Loaded", S.fname, C["mint"]), unsafe_allow_html=True)
        c[1].markdown(kpi("Sampling", f"{S.fs:g} Hz", f"{len(S.sig)} samples"), unsafe_allow_html=True)
        c[2].markdown(kpi("Duration", f"{duration():.1f} s", ""), unsafe_allow_html=True)
        c[3].markdown(kpi("Reference", "Clean" if S.has_ref else "None",
                          "SNR/MSE available" if S.has_ref else "SNR/MSE need a reference",
                          C["mint"] if S.has_ref else C["noise"]), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        left, right = st.columns([3, 2])
        with left:
            with st.container(border=True):
                st.markdown("##### Signal preview")
                noisy, _, err = compute_noisy()
                t = np.arange(len(S.sig)) / S.fs
                fig = base_fig(300, xrange=[0, min(duration(), 6)])
                if S.has_ref:
                    add_line(fig, t, S.sig, "Clean reference", C["mint"], 1.8)
                    add_line(fig, t, noisy, "Noisy ECG", C["noise"], 1.1, 0.75)
                else:
                    add_line(fig, t, S.sig, "Imported ECG (no reference)", C["noise"], 1.5)
                show(fig, "dash_prev")
        with right:
            with st.container(border=True):
                st.markdown("##### Latest filter results")
                R = S.results
                if R is None:
                    st.caption("No filter run yet.")
                    st.button("Go to Filter Comparison", on_click=goto, args=("Filter Comparison",),
                              type="primary", key="dash_go_filters")
                elif R["metrics"]:
                    best = max(R["metrics"], key=lambda k: R["metrics"][k]["snr_db"])
                    for k, _, short, col in FILTERS:
                        m = R["metrics"][k]
                        star = " ★" if k == best else ""
                        st.markdown(f'<div class="pl-kpi" style="margin-bottom:8px;border-left:4px solid {col}">'
                                    f'<div class="l">{short}{star}</div>'
                                    f'<div class="v" style="font-size:1.3rem">{fmt_snr(m["snr_db"])} dB'
                                    f'<span class="s"> · MSE {fmt_mse(m["mse"])}</span></div></div>',
                                    unsafe_allow_html=True)
                    if results_stale():
                        st.caption("Settings changed since this run — re-apply on Filter Comparison.")
                else:
                    st.markdown('<div class="pl-note">Filters applied. Reference required for SNR/MSE.</div>',
                                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("##### Processing pipeline")
        st.markdown(flow_html(), unsafe_allow_html=True)


def read_csv_numeric(upload):
    """Parse a CSV into a DataFrame; returns (df, numeric_columns)."""
    raw = upload.getvalue()
    if not raw.strip():
        raise ValueError("The file is empty.")
    first = raw.decode("utf-8", errors="ignore").lstrip().splitlines()[0]
    sep = next((c for c in (",", ";", "\t") if c in first), ",")
    try:
        df = pd.read_csv(io.BytesIO(raw), sep=sep)
    except Exception as e:
        raise ValueError(f"Could not read this file as CSV ({e}).")
    # header-less numeric file: first "column names" are numbers
    try:
        [float(str(c)) for c in df.columns]
        df = pd.read_csv(io.BytesIO(raw), sep=sep, header=None)
        df.columns = [f"column_{i}" for i in range(df.shape[1])]
    except ValueError:
        pass
    df.columns = [str(c).strip().strip("'\"") for c in df.columns]
    num = {}
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")  # also copes with a units row
        if s.notna().mean() >= 0.9 and s.notna().sum() >= 2:
            num[c] = s
    if not num:
        raise ValueError("No numeric columns found in this CSV.")
    return df, num


def page_import():
    st.markdown("### Import ECG")
    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            st.markdown("##### A. Import Clean Reference ECG")
            st.radio("This file contains", ["Clean reference ECG", "Noisy ECG (no clean reference)"],
                     key="import_mode", horizontal=True)
            if S.import_mode.startswith("Noisy"):
                st.markdown('<div class="pl-warn">Without a clean reference, filtering and heartbeat analysis '
                            'work, but reference-based SNR / MSE / RMSE <b>cannot be calculated reliably</b> '
                            'and will show “Reference required”.</div>', unsafe_allow_html=True)
            up = st.file_uploader("Drag and drop a CSV here", type=["csv", "txt"],
                                  label_visibility="collapsed", key="uploader")
            if up is not None:
                try:
                    df, num = read_csv_numeric(up)
                except ValueError as e:
                    st.error(str(e))
                    up = None
            if up is not None:
                cols = list(num.keys())
                tguess = next((c for c in cols if any(w in c.lower() for w in ("time", "elapsed", "sec"))), None)
                eguess = next((c for c in cols if c != tguess), cols[0])
                c1, c2, c3 = st.columns(3)
                ecg_col = c1.selectbox("ECG signal column", cols, index=cols.index(eguess))
                t_opts = ["(none)"] + [c for c in cols if c != ecg_col]
                tcol = c2.selectbox("Time column (optional)", t_opts,
                                    index=t_opts.index(tguess) if tguess in t_opts else 0)
                unit = c3.selectbox("Time unit", ["seconds", "milliseconds"], disabled=(tcol == "(none)"))
                est_fs = None
                if tcol != "(none)":
                    tv = num[tcol].dropna().to_numpy()
                    dt = np.median(np.diff(tv)) if len(tv) > 2 else 0
                    dt = dt / (1000.0 if unit == "milliseconds" else 1.0)
                    if dt > 0:
                        est_fs = 1.0 / dt
                S["fs_import"] = float(S.fs)
                c4, c5 = st.columns(2)
                c4.number_input("Sampling frequency (Hz)", min_value=1.0, max_value=100000.0, step=1.0,
                                key="fs_import", help="360 Hz for MIT-BIH data.")
                c5.number_input("Use first N seconds (0 = whole file)", min_value=0.0, step=1.0, key="seg_seconds")
                use_est = False
                if est_fs:
                    use_est = st.checkbox(f"Use sampling frequency estimated from the time column ({est_fs:.1f} Hz)", value=True)
                if st.button("Load this ECG", type="primary", key="load_csv"):
                    x = num[ecg_col].dropna().to_numpy(dtype=float)
                    fs = float(est_fs if use_est else S.fs_import)
                    if len(x) < 50:
                        st.error("The selected column has fewer than 50 valid samples.")
                    elif not np.all(np.isfinite(x)):
                        st.error("The selected column contains non-finite values.")
                    else:
                        if S.seg_seconds > 0:
                            x = x[: max(2, int(S.seg_seconds * fs))]
                        _reset_loaded(x, S.import_mode.startswith("Clean"), fs, up.name)
                        st.success(f"Loaded {len(x)} samples from “{up.name}”.")
    with right:
        with st.container(border=True):
            st.markdown("##### B. Load Sample ECG")
            st.caption("A synthetic 10 s, 360 Hz ECG-like signal (72 BPM). Perfect for a quick demo — "
                       "it is a clean reference, so SNR/MSE work immediately.")
            st.button("Load Sample ECG", on_click=load_sample, type="primary", key="imp_sample")
            sample_csv = pd.DataFrame({"time_s": np.arange(3600) / 360.0,
                                       "ecg_mV": dsp.generate_sample_ecg(360.0, 10.0)}).to_csv(index=False)
            st.download_button("Download sample as CSV", sample_csv, "pulselab_sample_ecg.csv", "text/csv",
                               on_click="ignore", key="dl_sample")
        with st.expander("Using real MIT-BIH record 109 (Colab export)"):
            st.code("import pandas as pd, wfdb\n"
                    "rec = wfdb.rdrecord('109', pn_dir='mitdb')\n"
                    "n = int(10 * rec.fs)\n"
                    "pd.DataFrame({'time_s': np.arange(n)/rec.fs,\n"
                    "              'ecg_mV': rec.p_signal[:n, 0]}).to_csv('ecg109.csv', index=False)",
                    language="python")
            st.caption("Upload `ecg109.csv` here as a **clean reference** (fs = 360 Hz).")

    st.markdown("<br>", unsafe_allow_html=True)
    if S.sig is not None:
        c = st.columns(5)
        c[0].markdown(kpi("File", "", S.fname), unsafe_allow_html=True)
        c[1].markdown(kpi("Samples", f"{len(S.sig):,}"), unsafe_allow_html=True)
        c[2].markdown(kpi("Sampling frequency", f"{S.fs:g} Hz"), unsafe_allow_html=True)
        c[3].markdown(kpi("Duration", f"{duration():.2f} s"), unsafe_allow_html=True)
        c[4].markdown(kpi("Status", "Clean reference" if S.has_ref else "Noisy input",
                          "reference preserved" if S.has_ref else "no reference — metrics off",
                          C["mint"] if S.has_ref else C["noise"]), unsafe_allow_html=True)
        with st.container(border=True):
            t = np.arange(len(S.sig)) / S.fs
            fig = base_fig(270)
            add_line(fig, t, S.sig, "Imported ECG", C["mint"], 1.5)
            show(fig, "imp_prev")


def page_noise():
    st.markdown("### Noise Simulation")
    if need_signal():
        return
    if not S.has_ref:
        st.markdown('<div class="pl-warn">The imported signal is already noisy and has no clean reference, '
                    'so noise simulation is disabled. It will be filtered as it is.</div>', unsafe_allow_html=True)
    ctl, plot = st.columns([1.1, 3])
    with ctl:
        with st.container(border=True):
            st.markdown("##### Controls")
            st.selectbox("Noise type", dsp.NOISE_KINDS, key="noise_kind", disabled=not S.has_ref)
            st.slider("Noise Intensity", 0, 100, key="noise_int", format="%d%%", disabled=not S.has_ref)
            st.markdown(f'<div class="pl-kpi"><div class="l">Current noise level</div>'
                        f'<div class="v" style="color:{C["noise"]}">{S.noise_int}%</div></div>',
                        unsafe_allow_html=True)
            st.number_input("Random seed", min_value=0, step=1, key="noise_seed", disabled=not S.has_ref,
                            help="Same seed = same noise every time (reproducible demos).")
            b1, b2 = st.columns(2)
            b1.button("Simulate Noise", on_click=new_noise_seed, type="primary", key="sim_noise",
                      disabled=not S.has_ref, help="Draws a fresh random noise realisation (new seed).")
            b2.button("Reset Noise", on_click=reset_noise, key="reset_noise", disabled=not S.has_ref)
            st.caption("Noise is added to a copy; the clean reference is never modified. "
                       "Amplitudes scale with the ECG's standard deviation.")
    noisy, noise, err = compute_noisy()
    if err:
        st.error(err)
    with plot:
        t = np.arange(len(S.sig)) / S.fs
        tabs = st.tabs(["Clean vs noisy ECG", "Noise component", "FFT spectrum"])
        with tabs[0]:
            fig = base_fig(400)
            if S.has_ref:
                add_line(fig, t, S.sig, "Clean reference", C["mint"], 2.0)
                add_line(fig, t, noisy, f"Noisy ECG ({S.noise_int}%)", C["noise"], 1.2, 0.85)
            else:
                add_line(fig, t, S.sig, "Imported (noisy) ECG", C["noise"], 1.4)
            show(fig, "noise_main")
        with tabs[1]:
            fig = base_fig(400)
            if noise is not None:
                add_line(fig, t, noise, "Added noise", C["noise"], 1.2)
            else:
                st.caption("No simulated noise for this signal.")
            show(fig, "noise_comp")
        with tabs[2]:
            fig = base_fig(400, xrange=[0, min(100, S.fs / 2)], ytitle="Magnitude")
            fig.update_xaxes(title="Frequency (Hz)")
            f, m = dsp.magnitude_spectrum(S.sig, S.fs)
            add_line(fig, f, m, "Clean reference" if S.has_ref else "Imported ECG", C["mint"], 1.6)
            if S.has_ref:
                f, m = dsp.magnitude_spectrum(noisy, S.fs)
                add_line(fig, f, m, "Noisy ECG", C["noise"], 1.2, 0.85)
            show(fig, "noise_fft")
            st.caption("Baseline wander appears near 0.3 Hz, power-line interference as a spike at 50 Hz, "
                       "Gaussian noise as a raised floor across all frequencies.")


def page_filters():
    st.markdown("### Filter Comparison")
    if need_signal():
        return
    noisy, _, err = compute_noisy()
    if err:
        st.error(err)

    top = st.columns([1.6, 3.4])
    with top[0]:
        st.button("Apply Filters & Compare", on_click=run_filters, type="primary", key="apply",
                  width="stretch")
    with top[1]:
        if S.run_errors:
            for e in S.run_errors:
                st.error(e)
        elif S.results is None:
            st.markdown('<div class="pl-note">Press <b>Apply Filters & Compare</b> to filter the noisy ECG '
                        'with all three filters.</div>', unsafe_allow_html=True)
        elif results_stale():
            st.markdown('<div class="pl-warn">Signal, noise or filter settings changed since the last run — '
                        'press <b>Apply Filters & Compare</b> again.</div>', unsafe_allow_html=True)

    with st.expander("Advanced Filter Settings"):
        S["fs_adv"] = float(S.fs)
        a = st.columns(5)
        a[0].number_input("Sampling frequency (Hz)", min_value=1.0, max_value=100000.0, step=1.0, key="fs_adv",
                          on_change=lambda: S.__setitem__("fs", float(S.fs_adv)))
        a[1].number_input("Lower cutoff (Hz)", min_value=0.01, step=0.1, format="%.2f", key="low_cut")
        a[2].number_input("Upper cutoff (Hz)", min_value=1.0, step=1.0, key="high_cut")
        a[3].number_input("FIR order", min_value=10, max_value=2000, step=2, key="fir_order",
                          help="Even order → odd number of taps (order + 1), linear phase.")
        a[4].number_input("Butterworth order", min_value=1, max_value=10, step=1, key="iir_order",
                          help="Band-pass of order N has 2N poles.")
        st.checkbox("Remove DC offset (mean) before computing SNR / MSE", key="remove_dc",
                    help="A 0.5 Hz high-pass cannot keep a constant offset, so it would count as error.")
        errs = dsp.validate_params(S.fs, S.low_cut, S.high_cut, S.fir_order, S.iir_order, len(S.sig))
        for e in errs:
            st.error(e)
        if not errs:
            st.caption(f"Nyquist frequency = {S.fs / 2:g} Hz. Defaults: 0.5–40 Hz keeps the ECG band "
                       f"and attenuates 50 Hz power-line noise.")

    p = get_params()
    specs = dsp.filter_specs(**p)
    R = S.results
    t = np.arange(len(S.sig)) / S.fs
    dur = duration()
    if S.view_win is None or S.view_win[1] > dur + 1e-9:
        S.view_win = (0.0, float(min(dur, 5.0)))
    if dur > 0.2:
        st.slider("View window (s) — applies to all graphs", 0.0, float(dur), step=0.1, key="view_win")
    xr = list(S.view_win)

    # common y range so the three cards are directly comparable
    pool = [S.sig, noisy] + (list(R["outputs"].values()) if R else [])
    ymin = min(float(np.min(a)) for a in pool)
    ymax = max(float(np.max(a)) for a in pool)
    pad = 0.08 * (ymax - ymin or 1)
    yr = [ymin - pad, ymax + pad]

    cols = st.columns(3)
    for col, (key, title, short, color) in zip(cols, FILTERS):
        sp = specs[key]
        with col:
            with st.container(border=True, key=f"card_{key}"):
                st.markdown(f'<div class="pl-card-title" style="color:{color}">{title}</div>',
                            unsafe_allow_html=True)
                fig = base_fig(270, xrange=xr, yrange=yr)
                if S.has_ref:
                    add_line(fig, t, S.sig, "Clean reference", C["mint"], 1.2, 0.8, "dot")
                else:
                    add_line(fig, t, noisy, "Input", C["noise"], 1.0, 0.5)
                if R:
                    add_line(fig, t[: len(R["outputs"][key])], R["outputs"][key], short, color, 1.8)
                show(fig, f"card_plot_{key}")
                if key == "butter":
                    items = [f"Order {sp['order']} (2·N = {2 * sp['order']} poles)",
                             f"Band {sp['low']:g}–{sp['high']:g} Hz", "Maximally flat passband",
                             "Zero-phase (filtfilt)"]
                else:
                    items = [f"Order {sp['order']} ({sp['taps']} taps)", f"Band {sp['low']:g}–{sp['high']:g} Hz",
                             f"{sp['window']} window", f"Transition ≈ {sp['transition_hz']:.1f} Hz",
                             f"Stop-band ≈ {sp['stopband_db']} dB", "Zero-phase (filtfilt)"]
                st.markdown(chips(items), unsafe_allow_html=True)
                if R and R["metrics"]:
                    m = R["metrics"][key]
                    st.markdown(f'<span style="color:{C["muted"]};font-size:.85rem">SNR '
                                f'<b style="color:{color}">{fmt_snr(m["snr_db"])} dB</b> · MSE '
                                f'<b style="color:{color}">{fmt_mse(m["mse"])}</b></span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tabs = st.tabs(["Combined comparison", "Filter performance", "Frequency response"])

    with tabs[0]:
        if R is None:
            st.caption("Apply the filters to compare all signals here.")
        else:
            t1 = st.columns(5)
            t1[0].checkbox("Original clean ECG", key="show_clean", disabled=not S.has_ref)
            t1[1].checkbox("Noisy ECG", key="show_noisy")
            t1[2].checkbox("FIR Hann", key="show_hann")
            t1[3].checkbox("FIR Hamming", key="show_hamming")
            t1[4].checkbox("IIR Butterworth", key="show_butter")
            fig = base_fig(430, xrange=xr, yrange=yr)
            fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.07, bgcolor=C["panel"]))
            n = len(R["noisy"])
            tt = np.arange(n) / R["params"]["fs"]
            if S.has_ref and S.show_clean:
                add_line(fig, tt, S.sig[:n], "Clean reference", C["mint"], 2.2)
            if S.show_noisy:
                add_line(fig, tt, R["noisy"], "Noisy ECG" if S.has_ref else "Input ECG", C["noise"], 1.0, 0.6)
            for k, _, short, col in FILTERS:
                if S.get(f"show_{k}"):
                    add_line(fig, tt, R["outputs"][k], short, col, 1.6)
            show(fig, "combined")
            st.caption("Drag to zoom, scroll to zoom, double-click to reset; use the slider below the plot "
                       "to scan through the recording.")

    with tabs[1]:
        st.markdown("##### Filter Performance")
        if R is None:
            st.caption("Apply the filters to see SNR and MSE.")
        elif not R["metrics"]:
            st.markdown('<div class="pl-warn"><b>Reference required.</b> The imported signal has no clean '
                        'reference, so SNR, MSE and RMSE cannot be calculated reliably. '
                        'Load a clean ECG (or the sample) to enable them.</div>', unsafe_allow_html=True)
            mc = st.columns(3)
            for c, (k, _, short, col) in zip(mc, FILTERS):
                c.markdown(kpi(short, "Reference required", "SNR / MSE unavailable", col), unsafe_allow_html=True)
        else:
            M = R["metrics"]
            best_snr = max(M, key=lambda k: M[k]["snr_db"])
            best_mse = min(M, key=lambda k: M[k]["mse"])
            mc = st.columns(3)
            for c, (k, _, short, col) in zip(mc, FILTERS):
                m = M[k]
                badge_s = f' <span class="pl-best">★ best</span>' if k == best_snr else ""
                badge_m = f' <span class="pl-best">★ best</span>' if k == best_mse else ""
                c.markdown(
                    f'<div style="color:{col};font-weight:700;margin-bottom:6px">{short}</div>'
                    f'<div style="display:flex;gap:10px">'
                    f'{kpi("SNR", fmt_snr(m["snr_db"]) + " dB", "higher is better" + badge_s, col)}'
                    f'{kpi("MSE", fmt_mse(m["mse"]), "RMSE " + fmt_mse(m["rmse"]) + badge_m, col)}</div>',
                    unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            rows = ""
            for k, _, short, col in FILTERS:
                m = M[k]
                cs = ' class="pl-best"' if k == best_snr else ""
                cm = ' class="pl-best"' if k == best_mse else ""
                rows += (f'<tr><td><span style="color:{col}">●</span> {short}</td>'
                         f'<td{cs}>{fmt_snr(m["snr_db"])}</td><td{cm}>{fmt_mse(m["mse"])}</td>'
                         f'<td{cm}>{fmt_mse(m["rmse"])}</td></tr>')
            st.markdown(f'<table class="pl-table"><tr><th>Filter</th><th>SNR (dB)</th><th>MSE (mV²)</th>'
                        f'<th>RMSE (mV)</th></tr>{rows}</table>', unsafe_allow_html=True)
            ch = st.columns(2)
            names = [short for _, _, short, _ in FILTERS]
            colors = [col for _, _, _, col in FILTERS]
            for c, (field, title, fmt) in zip(ch, [("snr_db", "SNR (dB) — higher is better", "{:.2f}"),
                                                    ("mse", "MSE (mV²) — lower is better", None)]):
                vals = [M[k][field] for k, *_ in FILTERS]
                f2 = go.Figure(go.Bar(x=names, y=vals, marker_color=colors,
                                      text=[fmt.format(v) if fmt else fmt_mse(v) for v in vals],
                                      textposition="outside", cliponaxis=False))
                f2.update_layout(height=300, title=dict(text=title, font=dict(size=14)),
                                 paper_bgcolor=C["card"], plot_bgcolor=C["plot"],
                                 font=dict(color=C["text"]), margin=dict(l=50, r=15, t=50, b=40),
                                 yaxis=dict(gridcolor=C["grid"]), xaxis=dict(gridcolor=C["grid"]))
                with c:
                    show(f2, f"bar_{field}")
            st.markdown(
                f'<div class="pl-note"><b>Convention.</b> x = clean reference, y = filtered output (same length). '
                f'MSE = (1/N)·Σ(x−y)², RMSE = √MSE, SNR = 10·log₁₀(Σx² / Σ(x−y)²). '
                f'Signal power is that of the clean reference; the “noise” is the whole residual error '
                f'(leftover noise + distortion). DC offset removed first: <b>{"yes" if R["remove_dc"] else "no"}</b>. '
                f'ECG samples are assumed to be in mV. Starred cells are the best in <i>this</i> run only — '
                f'results change with the noise type, intensity, cutoffs and orders. '
                f'Real recordings often contain baseline drift below the lower cutoff; a filter that removes it '
                f'is penalised because the reference still contains it.</div>', unsafe_allow_html=True)

    with tabs[2]:
        fr = dsp.frequency_responses(**p) if not dsp.validate_params(
            p["fs"], p["low"], p["high"], p["fir_order"], p["iir_order"], len(S.sig)) else None
        if fr is None:
            st.caption("Fix the parameter errors above to see the response.")
        else:
            fig = base_fig(380, xrange=[0, min(100, S.fs / 2)], yrange=[-100, 5], ytitle="Magnitude (dB)")
            fig.update_xaxes(title="Frequency (Hz)")
            for k, _, short, col in FILTERS:
                add_line(fig, fr[k][0], fr[k][1], short, col, 1.8)
            fig.add_vline(x=p["low"], line=dict(color=C["muted"], dash="dash", width=1))
            fig.add_vline(x=p["high"], line=dict(color=C["muted"], dash="dash", width=1))
            show(fig, "freq_resp")
            st.caption("Single-pass design responses. Dashed lines mark the cutoffs. Forward-backward filtering "
                       "applies the response twice (dB values double). The FIR transition width ≈ K·fs/taps, "
                       "so a short FIR cannot form a sharp 0.5 Hz high-pass edge — a Butterworth can.")


def page_heartbeat():
    st.markdown("### Heartbeat Analysis")
    st.markdown(f'<div style="color:{C["muted"]};margin-top:-8px">Preliminary ECG rhythm assessment using '
                f'detected R-peaks and RR intervals.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if need_signal():
        return
    R = S.results
    if R is None:
        st.markdown('<div class="pl-note">Heartbeat analysis uses a denoised signal. Apply the filters first.</div>',
                    unsafe_allow_html=True)
        st.button("Apply Filters & Compare", on_click=run_filters, type="primary", key="hb_run")
        if S.run_errors:
            for e in S.run_errors:
                st.error(e)
        return
    if results_stale():
        st.caption("⚠ Showing the last filter run; settings have changed since.")

    options = ["hann", "hamming", "butter", "input"]
    if S.hb_source not in options:
        S.hb_source = "hann"
    top = st.columns([2, 4])
    with top[0]:
        src = st.selectbox("Signal to analyse", options, key="hb_source",
                           format_func=lambda k: FTITLE[k] + " output" if k in FTITLE
                           else ("Noisy ECG (unfiltered)" if R["has_ref"] else "Imported ECG (unfiltered)"))
    with st.expander("Detection & rhythm thresholds (configurable)"):
        a = st.columns(4)
        a[0].number_input("RR variability limit — CV (%)", 1, 100, key="hb_cv",
                          help="SD(RR) / mean(RR). Above this → possible irregularity.")
        a[1].number_input("Beat-to-beat jump (%)", 1, 100, key="hb_jump",
                          help="A change in RR larger than this counts as a jump.")
        a[2].number_input("Max share of jumps (%)", 1, 100, key="hb_jump_frac")
        a[3].number_input("Min. beats needed", 3, 50, key="hb_min_beats")
        b = st.columns(4)
        b[0].number_input("Shortest plausible RR (ms)", 100, 1000, key="hb_min_rr")
        b[1].number_input("Longest plausible RR (ms)", 1000, 5000, key="hb_max_rr")
        b[2].number_input("Max implausible RR (%)", 0, 100, key="hb_bad_frac")
        b[3].number_input("Min. beat similarity (%)", 0, 100, key="hb_corr",
                          help="Quality check: detected beats must resemble each other.")
        st.caption("Heuristic: Insufficient quality if too short, too few beats, too many implausible RR "
                   "intervals, or beats do not look alike. Otherwise 'Possible irregularity' if CV or the "
                   "share of large jumps exceeds its limit; else 'Appears regular'.")

    x = R["noisy"] if src == "input" else R["outputs"][src]
    fs = R["params"]["fs"]
    res = hb.analyze_rhythm(x, fs, rhythm_cfg())

    c = st.columns(5)
    f1 = lambda v, d=0: "—" if v is None else f"{v:.{d}f}"
    c[0].markdown(kpi("Estimated heart rate", f1(res.hr_bpm), "BPM"), unsafe_allow_html=True)
    c[1].markdown(kpi("Detected R-peaks", str(len(res.peaks)), f"in {len(x) / fs:.1f} s"), unsafe_allow_html=True)
    c[2].markdown(kpi("Average RR interval", f1(res.mean_rr_ms), "ms"), unsafe_allow_html=True)
    c[3].markdown(kpi("RR interval variability",
                      "—" if res.cv is None else f"{res.cv * 100:.1f}%",
                      "—" if res.sd_rr_ms is None else f"SD {res.sd_rr_ms:.1f} ms · CV"), unsafe_allow_html=True)
    vcol = {hb.REGULAR: C["mint"], hb.IRREGULAR: C["noise"], hb.INSUFFICIENT: C["muted"]}[res.verdict]
    c[4].markdown(kpi("Rhythm regularity", res.verdict.replace("Possible Irregularity Detected", "Possible irregularity")
                      .replace("Insufficient Signal Quality", "Insufficient quality").replace("Appears Regular", "Regular"),
                      "", vcol), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([3.2, 1.6])
    with left:
        with st.container(border=True):
            t = np.arange(len(x)) / fs
            fig = base_fig(320)
            add_line(fig, t, x, "ECG", C["mint"], 1.5)
            if len(res.peaks):
                fig.add_trace(go.Scatter(x=res.peak_times_s, y=x[res.peaks], mode="markers", name="R-peaks",
                                         marker=dict(color=C["butter"], size=10, line=dict(color="#fff", width=1))))
            show(fig, "hb_ecg")
        with st.container(border=True):
            fig = base_fig(260, ytitle="RR interval (ms)")
            fig.update_xaxes(title="Time of beat (s)")
            if len(res.rr_ms):
                fig.add_trace(go.Scatter(x=res.rr_times_s, y=res.rr_ms, mode="lines+markers", name="RR interval",
                                         line=dict(color=C["violet"], width=2),
                                         marker=dict(color=np.where(res.rr_valid, C["violet"], C["noise"]).tolist(), size=8)))
                if res.mean_rr_ms:
                    fig.add_hline(y=res.mean_rr_ms, line=dict(color=C["cyan"], dash="dash", width=1.2),
                                  annotation_text=f"mean {res.mean_rr_ms:.0f} ms", annotation_font_color=C["cyan"])
                if (~res.rr_valid).any():
                    st.caption("Orange markers = RR intervals outside the plausible range (excluded from statistics).")
            else:
                st.caption("Not enough detected beats to plot RR intervals.")
            show(fig, "hb_rr")
    with right:
        with st.container(border=True):
            st.markdown("##### Result")
            st.markdown(f'<div class="pl-verdict" style="border-color:{vcol};background:{vcol}18">'
                        f'<div style="color:{vcol};font-weight:800;font-size:1.25rem">{res.verdict}</div></div>',
                        unsafe_allow_html=True)
            for r in res.reasons:
                st.markdown(f"- {r}")
            if res.rmssd_ms is not None:
                st.caption(f"RMSSD {res.rmssd_ms:.1f} ms · beat similarity "
                           f"{'—' if res.beat_corr is None else f'{res.beat_corr:.2f}'}")
    st.markdown('<div class="pl-note">Heartbeat regularity is estimated using signal-processing methods for '
                'educational purposes only. This tool cannot diagnose arrhythmias or replace professional '
                'medical evaluation.</div>', unsafe_allow_html=True)


def page_about():
    st.markdown("### About the Project")
    with st.container(border=True):
        st.markdown("PulseLab is an educational DSP application developed to study ECG signal denoising using "
                    "FIR filters with Hann and Hamming windows and an IIR Butterworth filter. The application "
                    "compares the performance of these filters using SNR, MSE, and RMSE, and demonstrates basic "
                    "heartbeat regularity analysis using R-peak detection and RR interval variability.")
    with st.container(border=True):
        st.markdown("##### Block diagram")
        st.markdown(flow_html(), unsafe_allow_html=True)
    c = st.columns(2)
    with c[0]:
        with st.container(border=True):
            st.markdown("##### Method notes")
            st.markdown(
                "- **Noise** (simulated, scaled to the ECG's std): 0.3 Hz baseline wander, 50 Hz power line, Gaussian.\n"
                "- **FIR**: window-method band-pass (`scipy.signal.firwin`), same order and cutoffs for Hann and Hamming.\n"
                "- **IIR**: Butterworth band-pass as second-order sections (`butter(..., output='sos')`).\n"
                "- **Zero-phase** filtering (`filtfilt` / `sosfiltfilt`): offline only, not real-time.\n"
                "- **Metrics**: MSE, RMSE, SNR = 10·log₁₀(Σx² / Σ(x−y)²) against the clean reference.\n"
                "- **Heartbeat**: 5–15 Hz band-pass → derivative² → moving average → peak picking → RR statistics.")
    with c[1]:
        with st.container(border=True):
            st.markdown("##### Code layout")
            st.markdown("- `app.py` — interface, state, graphs\n- `dsp.py` — filters, noise, metrics\n"
                        "- `heartbeat.py` — R-peaks, RR analysis, regularity heuristic")
            st.markdown("##### Limitations")
            st.markdown("Educational demo; simulated noise is not real recorded noise. Not clinically validated — "
                        "it cannot diagnose atrial fibrillation or any other arrhythmia.")


PAGE_FUNCS = {
    "Dashboard": page_dashboard, "Import ECG": page_import, "Noise Simulation": page_noise,
    "Filter Comparison": page_filters, "Heartbeat Analysis": page_heartbeat,
    "About the Project": page_about,
}

header()
PAGE_FUNCS[S.page]()
