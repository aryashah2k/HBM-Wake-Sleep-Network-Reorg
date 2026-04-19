# EEG-Phase Analysis Pipeline — ds003768

Phase-II analysis of the Sleep Network Reorganization project. All code
lives under `eeg/eeg_analysis/`; all paths are relative to `eeg/`.

## Cohort

32 subjects (sub-08 excluded, mirroring the fMRI paper), all
`task-rest` and `task-sleep` runs. Stages are read from the TSVs in
`eeg/sourcedata/` and mapped W→Wake, 1→N1, 2→N2.

## Pipeline stages

| # | Script | Output |
|---|--------|--------|
| 00 | `00_build_manifest.py` | `results/tables/manifest_runs_stages.csv` |
| 01 | `01_preprocess.py` | `derivatives/sub-XX/*_desc-aasbcg_raw.fif`, `*_desc-clean_raw.fif`, `*_provenance.json` |
| 02 | `02_stage_epochs.py` | `derivatives/sub-XX/*_desc-stages_epo.fif` |
| 03 | `03_psd_fooof.py` | `tables/fooof_params.csv`, `tables/fooof_occipital_summary.csv`, per-subject PSD figures |
| 04 | `04_alpha_wake_vs_n2.py` | `tables/alpha_power_by_stage.csv`, `tables/alpha_wake_vs_n2_stats.csv`, group figure |
| 05 | `05_bycycle_alpha.py` | `tables/bycycle_*.csv`, group violin figure |
| 06 | `06_multitaper.py` | per-run spectrograms (PNG+PDF), `tables/multitaper_band_power*.csv` |
| 07 | `07_dynamo_tfpeaks.py` | `tables/tfpeaks_all.csv.gz`, `tables/group_so_histograms.npz`, per-subject and group histograms |
| 08 | `08_group_stats_report.py` | `results/EEG_RESULTS.md` + `fig_eeg_summary.{png,pdf}` |

## Key algorithms

- **MR gradient artifact**: Allen (2000) sliding-window AAS using the
  `SyncStatus` volume triggers in `.vmrk`. Irregular spacing raises.
- **BCG**: Niazy (2005) Optimal Basis Set regression using
  R-peaks detected on the ECG channel.
- **ICA**: MNE extended Infomax (20 comps) with
  `find_bads_ecg` / `find_bads_eog` for automatic IC rejection.
- **Alpha Wake vs N2**: 8–13 Hz Welch band power, paired Wilcoxon with
  BH-FDR across occipital channels.
- **FOOOF**: per subject × stage × channel fit (1–40 Hz, fixed aperiodic).
- **bycycle**: cycle-by-cycle 7–14 Hz waveform shape (rdsym, ptsym,
  amp, period) on burst cycles only.
- **Multitaper**: faithful Python implementation of the Prerau Lab
  algorithm (DPSS tapers; NW=15, K=29; 30-s window, 5-s step).
- **DYNAM-O TF-peaks**: watershed segmentation on log-power
  spectrogram → peak features → SO-power and SO-phase histograms
  (Stokes et al., 2022).

## Running

```powershell
# From repo root
python -m pip install -r eeg\eeg_analysis\requirements.txt

python eeg\eeg_analysis\scripts\00_build_manifest.py
python eeg\eeg_analysis\scripts\01_preprocess.py --n-jobs 4
python eeg\eeg_analysis\scripts\02_stage_epochs.py
python eeg\eeg_analysis\scripts\03_psd_fooof.py
python eeg\eeg_analysis\scripts\04_alpha_wake_vs_n2.py
python eeg\eeg_analysis\scripts\05_bycycle_alpha.py
python eeg\eeg_analysis\scripts\06_multitaper.py
python eeg\eeg_analysis\scripts\07_dynamo_tfpeaks.py
python eeg\eeg_analysis\scripts\08_group_stats_report.py
```

Each script supports `--subjects sub-01 sub-02 …` and `--overwrite`.
Failures are logged to `eeg/logs/failures.csv` — nothing is silently
skipped at the group stage.

## Output layout

```
eeg/
  derivatives/<sub>/...fif, ...json, ...npz
  results/
    tables/*.csv(.gz), *.npz
    figures/
      png/group/*.png
      png/per_subject/<sub>/*.png
      pdf/…       (mirror of png)
    EEG_RESULTS.md
  logs/failures.csv
```
