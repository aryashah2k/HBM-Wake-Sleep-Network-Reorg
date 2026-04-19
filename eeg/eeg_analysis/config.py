"""Central configuration for the EEG-phase analysis pipeline.

All paths are resolved relative to the `eeg/` folder (the parent of
`eeg_analysis/`). No absolute paths are hard-coded.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EEG_ANALYSIS_DIR = Path(__file__).resolve().parent      # .../eeg/eeg_analysis
EEG_ROOT = EEG_ANALYSIS_DIR.parent                      # .../eeg
PROJECT_ROOT = EEG_ROOT.parent                          # .../sleep-eeg

RAW_DIR = EEG_ROOT                                      # subject folders live here
SOURCEDATA_DIR = EEG_ROOT / "sourcedata"                # sleep-stage TSVs

DERIVATIVES_DIR = EEG_ROOT / "derivatives"
RESULTS_DIR = EEG_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIG_DIR = RESULTS_DIR / "figures"
FIG_PNG_DIR = FIG_DIR / "png"
FIG_PDF_DIR = FIG_DIR / "pdf"
LOGS_DIR = EEG_ROOT / "logs"

for _d in (DERIVATIVES_DIR, RESULTS_DIR, TABLES_DIR,
           FIG_PNG_DIR, FIG_PDF_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Cohort
# ---------------------------------------------------------------------------
# Mirror the fMRI paper: 32 subjects, exclude sub-08.
ALL_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 34)]
EXCLUDED_SUBJECTS = {"sub-08"}
SUBJECTS = [s for s in ALL_SUBJECTS if s not in EXCLUDED_SUBJECTS]

# Stage code mapping (TSV strings -> canonical labels)
STAGE_MAP = {"W": "Wake", "0": "Wake",
             "1": "N1", "N1": "N1",
             "2": "N2", "N2": "N2"}
STAGES = ["Wake", "N1", "N2"]
STAGE_COLORS = {"Wake": "#1f77b4", "N1": "#ff7f0e", "N2": "#9467bd"}

EPOCH_DURATION_S = 30.0  # sleep staging epoch

# ---------------------------------------------------------------------------
# Acquisition / preprocessing
# ---------------------------------------------------------------------------
RAW_SFREQ = 5000.0          # Hz, from .vhdr
TARGET_SFREQ = 250.0        # Hz, after AAS

# Allen AAS parameters
AAS_WINDOW_VOLUMES = 25     # sliding template length
# In this BrainVision dataset, scanner volume triggers are `Response,R128`
# (TR = 2.1 s = 10500 samp at 5 kHz). `SyncStatus,Sync On` are clock
# heartbeats, not volume triggers.
AAS_MARKER_TYPE = "Response"
AAS_MARKER_DESCRIPTION = "R128"

# Filters
HPF_HZ = 0.3
LPF_HZ = 45.0
NOTCH_HZ = (50.0,)          # dataset origin (Siemens Prisma 3T); safe to apply

# ICA
ICA_N_COMPONENTS = 20
ICA_RANDOM_STATE = 97
ICA_METHOD = "infomax"
ICA_FIT_PARAMS = {"extended": True}

# Channel groups
OCCIPITAL_PRIMARY = ["O1", "O2", "Oz"]
OCCIPITAL_SECONDARY = ["Pz", "P3", "P4", "P7", "P8"]
OCCIPITAL_ALL = OCCIPITAL_PRIMARY + OCCIPITAL_SECONDARY
CENTRAL_SPINDLE = ["C3", "C4", "Cz"]
EOG_CH = "EOG"
ECG_CH = "ECG"

# Frequency bands
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (11.0, 16.0),
    "beta":  (16.0, 30.0),
}
ALPHA_BAND = BANDS["alpha"]

# PSD parameters
PSD_FMIN = 0.5
PSD_FMAX = 40.0
WELCH_WIN_SEC = 4.0
WELCH_OVERLAP = 0.5

# FOOOF
FOOOF_FREQ_RANGE = (1.0, 40.0)
FOOOF_PEAK_WIDTH_LIMITS = (1.0, 8.0)
FOOOF_MAX_N_PEAKS = 6
FOOOF_MIN_PEAK_HEIGHT = 0.1
FOOOF_PEAK_THRESHOLD = 2.0
FOOOF_APERIODIC_MODE = "fixed"

# bycycle
BYCYCLE_FREQ = (7.0, 14.0)
BYCYCLE_THRESHOLDS = dict(
    amp_fraction_threshold=0.3,
    amp_consistency_threshold=0.4,
    period_consistency_threshold=0.5,
    monotonicity_threshold=0.8,
    min_n_cycles=3,
)

# Multitaper (Prerau) — settings for visualization / stage band-power
# summaries (coarse time, heavy spectral smoothing).
MT_TIME_BANDWIDTH = 15.0
MT_NUM_TAPERS = 29
MT_WINDOW_S = 30.0
MT_STEP_S = 5.0
MT_FMIN = 0.5
MT_FMAX = 30.0

# TF-peak multitaper (fine time resolution, Stokes et al. 2022).
# A 0.05-s step is required so individual spindle-scale peaks
# (0.3–3 s duration) can be resolved.
MT_WINDOW_S_TFPEAK = 1.0
MT_STEP_S_TFPEAK = 0.05
MT_NW_TFPEAK = 2.0
MT_K_TFPEAK = 3

# DYNAM-O TF-peak extraction
TFPEAK_FREQ_RANGE = (9.0, 17.0)
TFPEAK_DUR_RANGE = (0.3, 3.0)
TFPEAK_BW_RANGE = (0.5, 4.0)
SO_BAND = (0.3, 1.5)
SO_POWER_BINS = 20           # for SO-power histogram
SO_PHASE_BINS = 20           # for SO-phase histogram
TFPEAK_FREQ_BINS = 40        # freq axis for histograms

# Epoch rejection. For 30-s epochs inside the MR scanner a 200 µV PTP
# threshold is too strict (drops ~every epoch). 500 µV is permissive
# enough to keep genuine EEG while still rejecting large movement
# artifacts; mastoids TP9/TP10 are excluded because residual BCG/
# electrode-pop there would otherwise kill otherwise-clean epochs.
EPOCH_PTP_UV = 500e-6
EPOCH_REJECT_EXCLUDE = ["TP9", "TP10"]
EPOCH_FLAT_FRAC = 0.3

# Reproducibility
GLOBAL_SEED = 42

# Plotting
FIG_DPI = 150


def png_path(name: str, subdir: str = "group") -> Path:
    p = FIG_PNG_DIR / subdir
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{name}.png"


def pdf_path(name: str, subdir: str = "group") -> Path:
    p = FIG_PDF_DIR / subdir
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{name}.pdf"


def subject_deriv_dir(subject: str) -> Path:
    d = DERIVATIVES_DIR / subject
    d.mkdir(parents=True, exist_ok=True)
    return d
