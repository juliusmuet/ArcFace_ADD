# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torchaudio
from pathlib import Path
import os
from tqdm import tqdm


def delete_broken_rirs(rir_dir):
    """
    Scan a directory for corrupted or unreadable Room Impulse Response (RIR) `.wav` files and delete them.

    This function recursively searches for all `.wav` files in the specified directory,
    attempts to load each using `torchaudio`, and deletes any files that fail to load
    (which are considered corrupted or unreadable).

    Args:
        rir_dir (str): The path to the directory containing the RIR `.wav` files to check.
    """
    rir_dir = Path(rir_dir)
    rir_files = list(rir_dir.glob("**/*.wav"))

    if not rir_files:
        print(f"No .wav files found in {rir_dir}")
        return

    print(f"Found {len(rir_files)} RIR files. Checking for corrupt or unreadable ones...\n")
    deleted = 0

    for rir_file in tqdm(rir_files, desc="Checking RIR files"):
        try:
            torchaudio.load(str(rir_file))
        except Exception as e:
            tqdm.write(f"Deleting broken file: {rir_file} | Error: {e}")
            try:
                os.remove(rir_file)
                deleted += 1
            except Exception as delete_error:
                tqdm.write(f"Failed to delete {rir_file}: {delete_error}")

    print(f"\nDone. Deleted {deleted} broken file(s) out of {len(rir_files)} total.")


if __name__ == "__main__":
    delete_broken_rirs('.../augmentation_data/RIRS_NOISES')
