import os
import glob
import datalad.api as dl  # make sure `datalad` is installed in this Python env

# Path to the root of your dataset
DATASET_ROOT = r"C:\Users\Arya\Desktop\sleep-eeg\ds003768"

def main():
    # Create a Dataset object for the root
    ds = dl.Dataset(DATASET_ROOT)
    if not ds.is_installed():
        raise RuntimeError(f"No DataLad dataset found at {DATASET_ROOT}")

    # Find all sub-XX/eeg folders (non-recursive under dataset root)
    pattern = os.path.join(DATASET_ROOT, "sub-*", "eeg")
    eeg_dirs = sorted(glob.glob(pattern))

    if not eeg_dirs:
        print("No sub-*/eeg directories found.")
        return

    print(f"Found {len(eeg_dirs)} eeg folders:")
    for d in eeg_dirs:
        print("  ", d)

    # Download content of each eeg directory
    for eeg_dir in eeg_dirs:
        rel_path = os.path.relpath(eeg_dir, DATASET_ROOT)
        print(f"\nGetting data for: {rel_path}")
        # ds.get works on paths relative to the dataset root
        ds.get(path=rel_path)  # gets all files in that folder [web:1][web:2]

if __name__ == "__main__":
    main()
