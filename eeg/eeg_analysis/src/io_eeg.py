"""BrainVision I/O and run enumeration."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import mne

from eeg_analysis import config as cfg

_RUN_RE = re.compile(r"(sub-\d+)_task-(rest|sleep)_run-(\d+)_eeg\.vhdr$")


@dataclass(frozen=True)
class RunInfo:
    subject: str
    task: str        # "rest" or "sleep"
    run: int
    vhdr: Path

    @property
    def session_label(self) -> str:
        """Matches the `session` column in sourcedata TSV files."""
        return f"task-{self.task}_run-{self.run}"

    @property
    def tag(self) -> str:
        return f"{self.subject}_task-{self.task}_run-{self.run}"


def list_runs(subject: str) -> List[RunInfo]:
    """Enumerate all VHDR runs for a subject, sorted."""
    eeg_dir = cfg.RAW_DIR / subject / "eeg"
    if not eeg_dir.is_dir():
        return []
    runs: List[RunInfo] = []
    for vhdr in sorted(eeg_dir.glob(f"{subject}_task-*_run-*_eeg.vhdr")):
        m = _RUN_RE.search(vhdr.name)
        if not m:
            continue
        runs.append(RunInfo(subject=m.group(1), task=m.group(2),
                            run=int(m.group(3)), vhdr=vhdr))
    return runs


def list_all_runs(subjects: List[str] | None = None) -> List[RunInfo]:
    subjects = subjects if subjects is not None else cfg.SUBJECTS
    out: List[RunInfo] = []
    for s in subjects:
        out.extend(list_runs(s))
    return out


def _ensure_vhdr_references_valid(vhdr: Path) -> Path:
    """Return a vhdr path whose DataFile/MarkerFile references exist.

    Some files in this dataset declare e.g. `DataFile=rsub-XX_...eeg`
    while the on-disk file is `sub-XX_...eeg`. When that happens we
    write a patched vhdr/vmrk pair into a cache folder and return the
    patched vhdr path; the originals are left untouched.
    """
    eeg_dir = vhdr.parent
    text = vhdr.read_text(encoding="latin-1")
    data_line = next((l for l in text.splitlines()
                      if l.startswith("DataFile=")), None)
    mrk_line = next((l for l in text.splitlines()
                     if l.startswith("MarkerFile=")), None)
    if data_line is None:
        return vhdr
    declared_data = data_line.split("=", 1)[1].strip()
    declared_mrk = mrk_line.split("=", 1)[1].strip() if mrk_line else None
    # Resolve relative references
    data_exists = (eeg_dir / declared_data).is_file()
    mrk_exists = (declared_mrk is None) or (eeg_dir / declared_mrk).is_file()
    if data_exists and mrk_exists:
        return vhdr
    # Expected corrected names (based on the vhdr stem)
    fixed_data = vhdr.with_suffix(".eeg").name
    fixed_mrk = vhdr.with_suffix(".vmrk").name
    if not (eeg_dir / fixed_data).is_file():
        raise FileNotFoundError(
            f"Cannot locate a valid data file for {vhdr.name}: "
            f"declared '{declared_data}' missing and fallback "
            f"'{fixed_data}' also missing.")
    # Write patched vhdr + matching vmrk into a cache folder
    cache = vhdr.parent / ".vhdr_patched"
    cache.mkdir(exist_ok=True)
    patched_vhdr = cache / vhdr.name
    new_text = text.replace(f"DataFile={declared_data}",
                            f"DataFile={fixed_data}")
    if declared_mrk is not None:
        new_text = new_text.replace(f"MarkerFile={declared_mrk}",
                                    f"MarkerFile={fixed_mrk}")
    patched_vhdr.write_text(new_text, encoding="latin-1")
    # The vmrk is referenced by filename only; copy the actual on-disk
    # vmrk into the cache under the expected name so MNE finds it.
    src_mrk = eeg_dir / fixed_mrk
    if src_mrk.is_file():
        (cache / fixed_mrk).write_bytes(src_mrk.read_bytes())
    # Also symlink/copy the .eeg file reference target
    # (MNE resolves DataFile relative to the vhdr's folder)
    target_eeg = cache / fixed_data
    if not target_eeg.exists():
        # On Windows, hardlink if possible, else copy
        try:
            import os
            os.link(eeg_dir / fixed_data, target_eeg)
        except OSError:
            target_eeg.write_bytes((eeg_dir / fixed_data).read_bytes())
    return patched_vhdr


def load_raw(run: RunInfo, preload: bool = True) -> mne.io.BaseRaw:
    """Load a BrainVision run and set channel types / montage."""
    vhdr = _ensure_vhdr_references_valid(run.vhdr)
    raw = mne.io.read_raw_brainvision(str(vhdr), preload=preload,
                                      verbose="ERROR")
    # Channel types
    ch_types = {}
    if cfg.EOG_CH in raw.ch_names:
        ch_types[cfg.EOG_CH] = "eog"
    if cfg.ECG_CH in raw.ch_names:
        ch_types[cfg.ECG_CH] = "ecg"
    if ch_types:
        raw.set_channel_types(ch_types, verbose="ERROR")

    # Standard 10-20 montage for EEG channels that match
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False, on_missing="ignore",
                        verbose="ERROR")
    except Exception:
        # Montage is optional for computations; we never silently
        # swallow errors that would affect results (topomaps will warn).
        pass
    return raw


def scanner_triggers(run: RunInfo) -> list[int]:
    """Read scanner volume triggers from the .vmrk file.

    Returns sorted, deduplicated sample indices (int) of every marker
    whose type matches `cfg.AAS_MARKER_TYPE` and (if configured)
    whose description matches `cfg.AAS_MARKER_DESCRIPTION`.
    """
    vmrk = run.vhdr.with_suffix(".vmrk")
    if not vmrk.is_file():
        raise FileNotFoundError(f"Marker file not found: {vmrk}")
    want_type = cfg.AAS_MARKER_TYPE
    want_desc = getattr(cfg, "AAS_MARKER_DESCRIPTION", None)
    triggers: list[int] = []
    # .vmrk is text with non-UTF-8 chars possible; use latin-1 safe decode
    with open(vmrk, "r", encoding="latin-1") as f:
        for line in f:
            if not line.startswith("Mk"):
                continue
            try:
                _, rhs = line.split("=", 1)
            except ValueError:
                continue
            parts = rhs.strip().split(",")
            if len(parts) < 3:
                continue
            mtype = parts[0].strip()
            mdesc = parts[1].strip()
            if mtype != want_type:
                continue
            if want_desc is not None and mdesc != want_desc:
                continue
            try:
                triggers.append(int(parts[2]))
            except ValueError:
                continue
    return sorted(set(triggers))
