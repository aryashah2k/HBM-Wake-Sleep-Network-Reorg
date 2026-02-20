"""
Central configuration for the Sleep Network Reorganization Analysis Pipeline.

All parameters are literature-backed with citations provided in comments.
Dataset: ds003768 — "Simultaneous EEG and fMRI signals during sleep from humans"
Preprocessed with fMRIPrep 23.0.2.
"""

import os
from pathlib import Path

# ==============================================================================
# Directory Paths
# ==============================================================================

PROJECT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREPROCESSED_DIR = PROJECT_DIR / "preprocessed"
SOURCEDATA_DIR = PREPROCESSED_DIR / "sourcedata"
RESULTS_DIR = PROJECT_DIR / "results"

# Output subdirectories
QC_DIR = RESULTS_DIR / "qc"
GLM_DIR = RESULTS_DIR / "glm"
TIMESERIES_DIR = RESULTS_DIR / "timeseries"
DFC_DIR = RESULTS_DIR / "dfc"
THALAMOCORTICAL_DIR = RESULTS_DIR / "thalamocortical"

FIGURES_DIR = RESULTS_DIR / "figures"
REPORT_DIR = RESULTS_DIR / "report"
RELIABILITY_DIR = RESULTS_DIR / "reliability"

ALL_OUTPUT_DIRS = [
    QC_DIR, GLM_DIR, TIMESERIES_DIR, DFC_DIR,
    THALAMOCORTICAL_DIR, FIGURES_DIR, REPORT_DIR, RELIABILITY_DIR,
]

# ==============================================================================
# Subject List
# ==============================================================================

SUBJECTS = [f"sub-{i:02d}" for i in range(1, 34)]  # sub-01 to sub-33

# ==============================================================================
# Acquisition Parameters (from dataset metadata)
# ==============================================================================

TR = 2.1  # Repetition time in seconds (from BOLD json sidecar)
EPOCH_DURATION = 30  # Sleep-stage scoring epoch duration in seconds
TRS_PER_EPOCH = int(EPOCH_DURATION / TR)  # ~14 TRs per 30-sec epoch

# ==============================================================================
# Sleep Stage Mapping
# ==============================================================================

STAGE_LABELS = {
    "W": "Wake",
    "1": "N1",
    "2": "N2",
}

# Stages to include in analysis (exclude "Unscorable", "uncertain", "3" if present)
VALID_STAGES = {"W", "1", "2"}

# Numeric encoding for GLM and statistics
STAGE_NUMERIC = {"W": 0, "1": 1, "2": 2}

# ==============================================================================
# Parcellation Settings
# ==============================================================================

# Schaefer 2018 atlas: 200 cortical parcels, 7-network annotation
# Reference: Schaefer et al., "Local-Global Parcellation of the Human Cerebral
#   Cortex from Intrinsic Functional Connectivity MRI", Cerebral Cortex, 2018.
ATLAS_NAME = "schaefer_2018"
N_PARCELS = 200
N_NETWORKS = 7
ATLAS_RESOLUTION_MM = 2  # Matching MNI152NLin2009cAsym res-2

# Yeo 7-network names (canonical ordering from Schaefer atlas labels)
YEO7_NETWORKS = [
    "Vis",       # Visual
    "SomMot",    # Somatomotor
    "DorsAttn",  # Dorsal Attention
    "SalVentAttn",  # Salience / Ventral Attention
    "Limbic",    # Limbic
    "Cont",      # Frontoparietal Control
    "Default",   # Default Mode
]

YEO7_NETWORK_FULL_NAMES = {
    "Vis": "Visual",
    "SomMot": "Somatomotor",
    "DorsAttn": "Dorsal Attention",
    "SalVentAttn": "Salience/Ventral Attention",
    "Limbic": "Limbic",
    "Cont": "Frontoparietal Control",
    "Default": "Default Mode",
}

# ==============================================================================
# Thalamic ROI Definitions (FreeSurfer aseg labels)
# ==============================================================================

# Reference: FreeSurfer LUT — https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
THALAMUS_LABELS = {
    "Left-Thalamus": 10,
    "Right-Thalamus": 49,
}

# ==============================================================================
# Confound Regression Strategy
# ==============================================================================

# 24-parameter motion model: 6 rigid-body params + their temporal derivatives
#   + quadratic terms of both (Friston et al., 1996; Satterthwaite et al., 2013)
# Reference: Satterthwaite et al., "An improved framework for confound regression
#   and filtering for control of motion artifact in the preprocessing of
#   resting-state functional connectivity data", NeuroImage, 2013.
MOTION_CONFOUNDS = [
    "trans_x", "trans_y", "trans_z",
    "rot_x", "rot_y", "rot_z",
    "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
    "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1",
    "trans_x_power2", "trans_y_power2", "trans_z_power2",
    "rot_x_power2", "rot_y_power2", "rot_z_power2",
    "trans_x_derivative1_power2", "trans_y_derivative1_power2", "trans_z_derivative1_power2",
    "rot_x_derivative1_power2", "rot_y_derivative1_power2", "rot_z_derivative1_power2",
]

# aCompCor: top 5 components from combined WM+CSF mask
# Reference: Behzadi et al., "A component based noise correction method (CompCor)
#   for BOLD and perfusion based fMRI", NeuroImage, 2007.
N_ACOMPCOR = 5
ACOMPCOR_CONFOUNDS = [f"a_comp_cor_{i:02d}" for i in range(N_ACOMPCOR)]

ALL_CONFOUNDS = MOTION_CONFOUNDS + ACOMPCOR_CONFOUNDS

# ==============================================================================
# Motion Censoring (Scrubbing)
# ==============================================================================

# FD threshold for volume censoring
# Reference: Power et al., "Methods to detect, characterize, and remove motion
#   artifact in resting state fMRI", NeuroImage, 2014.
# 0.3 mm is the stringent threshold recommended for resting-state/sleep fMRI.
# More conservative than the original 0.5 mm (Power et al., 2012) to control
# for subtle micromovements that affect functional connectivity.
FD_THRESHOLD = 0.3  # mm

# Subjects with > MAX_CENSORED_FRACTION of volumes censored are excluded.
# Reference: Power et al., 2014 — 50% threshold commonly used.
MAX_CENSORED_FRACTION = 0.50

# Minimum number of low-motion TRs required per stage for a run to be included
MIN_TRS_PER_STAGE = 30  # ~63 seconds minimum per stage

# ==============================================================================
# Bandpass Filter Settings
# ==============================================================================

# Standard resting-state fMRI bandpass filter range
# Reference: Biswal et al., 1995; Power et al., 2014
HIGH_PASS = 0.01  # Hz
LOW_PASS = 0.1    # Hz

# ==============================================================================
# GLM Settings
# ==============================================================================

# High-pass filter cutoff for GLM (standard FSL/SPM convention)
GLM_HIGH_PASS_PERIOD = 128  # seconds (1/128 Hz = 0.0078 Hz)

# Cluster-forming threshold for group-level analysis
# Reference: Eklund et al., "Cluster failure: Why fMRI inferences for spatial
#   extent have inflated false-positive rates", PNAS, 2016.
GLM_VOXEL_THRESHOLD = 0.001  # uncorrected p-value for cluster-forming
GLM_CLUSTER_ALPHA = 0.05     # FWE-corrected cluster-level significance

# ==============================================================================
# Dynamic Functional Connectivity (dFC) Settings
# ==============================================================================

# Sliding window size
# Reference: Leonardi & Van De Ville, "On spurious and real fluctuations of
#   dynamic functional connectivity during rest", NeuroImage, 2015.
# Recommended minimum: 1/f_min where f_min is the lowest frequency of interest.
# With bandpass 0.01 Hz → minimum ~100s. However, since we bandpass-filter
# during extraction (Step 2), shorter windows are acceptable.
# We use 30 TRs (63s ≈ 2 sleep epochs) to minimize stage mixing within
# windows while maintaining sufficient frequency content.
# Additional references: Preti et al., "The dynamic functional connectome:
#   State-of-the-art and perspectives", NeuroImage, 2017.
DFC_WINDOW_SIZE = 30  # TRs (~63 seconds, ~2 sleep epochs)
DFC_STEP_SIZE = 15    # TRs (~31.5s, ~1 epoch step to reduce autocorrelation)

# Graph thresholding: proportional threshold to retain top edges
# Reference: Rubinov & Sporns, "Complex network measures of brain connectivity:
#   Uses and interpretations", NeuroImage, 2010.
GRAPH_DENSITY_THRESHOLD = 0.15  # Retain top 15% of connections

# Louvain modularity resolution parameter
# Reference: Reichardt & Bornholdt, 2006; Rubinov & Sporns, 2011
MODULARITY_GAMMA = 1.0  # Standard resolution

# Number of Louvain iterations for stability
MODULARITY_N_ITER = 100

# ==============================================================================
# Thalamocortical Coupling Settings
# ==============================================================================

# Minimum contiguous block length for stage-specific connectivity
# Ensures stable correlation estimates within each sleep-stage block.
MIN_BLOCK_LENGTH_TRS = 10  # TRs (~21 seconds)

# ==============================================================================

# ==============================================================================
# Test-Retest Reliability Settings
# ==============================================================================

# ICC type for test-retest reliability
# Reference: Shrout & Fleiss, "Intraclass correlations: Uses in assessing rater
#   reliability", Psychological Bulletin, 1979.
# ICC(3,1) — two-way mixed, single measures, consistency
ICC_TYPE = "ICC3"

# Split-half reliability: number of random splits
N_SPLIT_HALF = 100

# Bootstrap iterations for confidence intervals
N_BOOTSTRAP = 1000
BOOTSTRAP_CI = 0.95

# ==============================================================================
# Visualization Settings
# ==============================================================================

FIGURE_DPI = 300
FIGURE_FORMAT = ["png", "pdf"]  # Save in both formats for publication

# Color palette for sleep stages (colorblind-friendly)
STAGE_COLORS = {
    "Wake": "#2196F3",   # Blue
    "N1": "#FF9800",     # Orange
    "N2": "#9C27B0",     # Purple
}

# Matplotlib settings for publication quality
FONT_FAMILY = "Arial"
FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 12
FONT_SIZE_TICK = 10
FONT_SIZE_LEGEND = 10

# Colormap for connectivity matrices
CONNECTIVITY_CMAP = "RdBu_r"

# ==============================================================================
# Statistical Settings
# ==============================================================================

# Significance level
ALPHA = 0.05

# Multiple comparisons correction method
# Reference: Benjamini & Hochberg, "Controlling the false discovery rate:
#   a practical and powerful approach to multiple testing", JRSS-B, 1995.
FDR_METHOD = "fdr_bh"

# ==============================================================================
# Utility Functions
# ==============================================================================


def setup_directories():
    """Create all output directories if they do not exist."""
    for d in ALL_OUTPUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def get_bold_path(subject, task, run, space="MNI152NLin2009cAsym", res="2"):
    """Return the path to the preprocessed BOLD file for a given subject/task/run."""
    filename = (
        f"{subject}_{task}_{run}_space-{space}_res-{res}_desc-preproc_bold.nii.gz"
    )
    return PREPROCESSED_DIR / subject / "func" / filename


def get_confounds_path(subject, task, run):
    """Return the path to the confounds TSV for a given subject/task/run."""
    filename = f"{subject}_{task}_{run}_desc-confounds_timeseries.tsv"
    return PREPROCESSED_DIR / subject / "func" / filename


def get_brain_mask_path(subject, task, run, space="MNI152NLin2009cAsym", res="2"):
    """Return the path to the MNI-space brain mask for a given subject/task/run."""
    filename = (
        f"{subject}_{task}_{run}_space-{space}_res-{res}_desc-brain_mask.nii.gz"
    )
    return PREPROCESSED_DIR / subject / "func" / filename


def get_aparc_path(subject, task, run, space="MNI152NLin2009cAsym", res="2"):
    """Return the path to the aparcaseg segmentation in MNI space."""
    filename = (
        f"{subject}_{task}_{run}_space-{space}_res-{res}_desc-aparcaseg_dseg.nii.gz"
    )
    return PREPROCESSED_DIR / subject / "func" / filename


def get_sleep_stage_path(subject):
    """Return the path to the sleep-stage TSV in sourcedata."""
    return SOURCEDATA_DIR / f"{subject}-sleep-stage.tsv"


def get_run_ids(subject):
    """
    Discover all available task/run combinations for a subject
    by scanning the func directory for preprocessed BOLD files.
    """
    func_dir = PREPROCESSED_DIR / subject / "func"
    if not func_dir.exists():
        return []

    runs = []
    import re
    pattern = re.compile(
        rf"{re.escape(subject)}_(?P<task>task-\w+)_(?P<run>run-\d+)_"
        r"space-MNI152NLin2009cAsym_res-2_desc-preproc_bold\.nii\.gz"
    )
    for f in func_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            runs.append((m.group("task"), m.group("run")))
    return sorted(runs)
