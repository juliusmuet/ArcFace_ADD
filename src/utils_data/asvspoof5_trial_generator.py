# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import random
from pathlib import Path


def load_references(reference_file):
    """
    Load the reference utterances from the reference file into a dictionary.

    Args:
        reference_file (str or Path): Path to the reference file.

    Returns:
        dict: A dictionary mapping speaker IDs to a list of their reference utterances.
              Example: { "D_4205": ["D_A0000000562", "D_A0000000898", ...], ... }
    """
    references = {}
    reference_file = Path(reference_file)
    with reference_file.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            speaker_id = parts[0]
            utterances = parts[1].split(",")
            references[speaker_id] = utterances
    return references


def process_trials(trial_file, references, audio_path, output_file):
    """
    Process the trial file and generate a modified output file with:
    reference utterance path, suspect utterance path and trial outcome.

    Args:
        trial_file (str or Path): Path to the trial file.
        references (dict): Dictionary mapping speaker IDs to their reference utterances.
        audio_path (str or Path): Base path where audio utterances are stored.
        output_file (str or Path): Path where the modified trial file will be written.
    """
    trial_file = Path(trial_file)
    audio_path = Path(audio_path)
    output_file = Path(output_file)

    with trial_file.open("r") as fin, output_file.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            speaker_id = parts[0]
            suspect_utt = parts[1]
            outcome = parts[-1]

            # Randomly select a reference utterance for this speaker
            if speaker_id not in references:
                raise ValueError(f"No reference utterances found for speaker ID: {speaker_id}")
            reference_utt = random.choice(references[speaker_id])

            # Construct full paths using pathlib
            reference_path = audio_path / f"{reference_utt}.wav"
            suspect_path = audio_path / f"{suspect_utt}.wav"

            # Write formatted line to output
            fout.write(f"{reference_path},{suspect_path},{outcome}\n")


def main():
    reference_file = r"...\ASVspoof5_protocols\ASVspoof5.dev.track_2.enroll.tsv"
    trial_file = r"...\ASVspoof5_protocols\ASVspoof5.dev.track_2.trial.tsv"
    output_file = r"...\ASVspoof5_protocols\dev_trials.txt"
    audio_path = "/path/to/audio/files"

    # Set fixed random seed for reproducibility
    random.seed(42)

    # Load reference data
    references = load_references(reference_file)

    # Process trials and generate output
    process_trials(trial_file, references, audio_path, output_file)


if __name__ == "__main__":
    main()
