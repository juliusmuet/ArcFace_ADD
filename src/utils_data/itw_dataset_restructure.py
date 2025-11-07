# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from pathlib import Path
import shutil
from tqdm import tqdm


# Define original and output dataset directories
ROOT_DIR = Path("path_to_your_dataset_folder")               # Original dataset
OUTPUT_ROOT_DIR = Path("path_to_output_dataset_folder")      # New restructured dataset

ORIGINAL_LABELS_FILE = ROOT_DIR / "l_original.txt"
OUTPUT_LABELS_FILE = OUTPUT_ROOT_DIR / "l_original.txt"

TEST_FILE = ROOT_DIR / "labels_fraunhofer_in_the_wild-test.txt"
OUTPUT_TEST_FILE = OUTPUT_ROOT_DIR / "labels_fraunhofer_in_the_wild-test.txt"

TRAIN_FILE = ROOT_DIR / "labels_fraunhofer_in_the_wild-train.txt"
OUTPUT_TRAIN_FILE = OUTPUT_ROOT_DIR / "labels_fraunhofer_in_the_wild-train.txt"


def restructure_files_and_update_l_original():
    """
    Reads the 'l_original.txt' file from the original dataset and:
    
    - Copies .wav files into a nested structure in the output directory: label/speaker/filename.wav,
      where 'label' is 'bonafide' or 'spoof', and 'speaker' uses underscores instead of spaces.
    - Updates 'bona-fide' to 'bonafide' in labels.
    - Writes a new 'l_original.txt' file to the output directory with updated paths and standardised labels.
    """
    updated_lines = []

    lines = ORIGINAL_LABELS_FILE.read_text().splitlines()

    for line in tqdm(lines, desc="Restructuring l_original.txt"):
        line = line.strip()
        if not line:
            continue
        wav_file, speaker, label = line.split(',')

        # Normalise values
        speaker = speaker.strip()
        speaker_dir = speaker.replace(" ", "_")
        label_dir = label.replace("bona-fide", "bonafide")

        # Create new directory path
        new_dir = OUTPUT_ROOT_DIR / label_dir / speaker_dir
        new_dir.mkdir(parents=True, exist_ok=True)

        # Copy file
        src_path = ROOT_DIR / wav_file
        dst_path = new_dir / Path(wav_file).name
        shutil.copy(src_path, dst_path)

        # Update label entry
        relative_path = dst_path.relative_to(OUTPUT_ROOT_DIR)
        updated_line = f"{relative_path},{speaker},{label_dir}"
        updated_lines.append(updated_line)

    # Write updated label file
    OUTPUT_LABELS_FILE.write_text('\n'.join(updated_lines) + '\n')


def update_secondary_labels_file(file_path: Path, output_path: Path):
    """
    Updates paths in a secondary label file (test or train) from the original dataset
    and writes the updated version to the output directory.

    Args:
        file_path : str
            Path to the original secondary label file.
        output_path : str
            Path to write the updated file.
    """
    updated_lines = []

    lines = file_path.read_text().splitlines()

    for line in tqdm(lines, desc=f"Updating {file_path.name}"):
        parts = line.strip().split(',')
        if len(parts) != 5:
            continue
        wav_file, _, label, _, speaker = parts

        # Normalise
        speaker = speaker.strip()
        speaker_dir = speaker.replace(" ", "_")
        label_dir = label.replace("bona-fide", "bonafide")

        # New path relative to OUTPUT_ROOT_DIR
        new_path = Path(label_dir) / speaker_dir / wav_file

        parts[0] = str(new_path)
        parts[2] = label_dir

        updated_lines.append(','.join(parts))

    output_path.write_text('\n'.join(updated_lines) + '\n')


if __name__ == "__main__":
    OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    restructure_files_and_update_l_original()
    update_secondary_labels_file(TEST_FILE, OUTPUT_TEST_FILE)
    update_secondary_labels_file(TRAIN_FILE, OUTPUT_TRAIN_FILE)

    print("Restructuring complete. Original dataset remains unchanged.")
