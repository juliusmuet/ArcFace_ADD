# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import random
import argparse
from collections import defaultdict, Counter
from typing import List
from tqdm import tqdm


def load_samples(file_path: str):
    """
    Load samples from the input file and group them by speaker and label.

    The file must have lines formatted as: path,speaker,label

    Args:
        file_path (str): Path to the input file.

    Returns:
        tuple:
            - bonafide_samples (dict): {speaker: list of bonafide file paths}
            - spoofed_samples (dict): {speaker: list of spoof file paths}
    """
    bonafide_samples = defaultdict(list)
    spoofed_samples = defaultdict(list)

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            path = parts[0].strip()
            speaker = parts[1].strip()
            label = parts[2].strip().lower()

            if label == "bonafide":
                bonafide_samples[speaker].append(path)
            elif label == "spoof":
                spoofed_samples[speaker].append(path)

    return bonafide_samples, spoofed_samples


def select_random_least_used(paths: List[str], usage_counter: Counter, top_n=5):
    """
    Select a path from the least-used paths, randomly among the top N least used.

    Args:
        paths (List[str]): List of file paths.
        usage_counter (Counter): Tracks how often each path has been used.
        top_n (int): Consider the top N least-used paths.

    Returns:
        str: Selected path.
    """
    sorted_paths = sorted(paths, key=lambda x: usage_counter[x])
    candidates = sorted_paths[:min(top_n, len(sorted_paths))]
    return random.choice(candidates)


def create_target_trials(bonafide_samples, n, usage_counter, max_usage, min_per_speaker=0):
    """
    Generate target trials: both audio samples are bonafide and from the same speaker.

    This function ensures that a minimum number of trials per speaker is generated,
    even if that means exceeding the max_usage limit for some samples.

    Args:
        bonafide_samples (dict): {speaker_id: list of bonafide file paths}
        n (int): Total number of target trials to generate.
        usage_counter (Counter): Tracks how often each sample has been used.
        max_usage (int): Maximum allowed usage for any sample (can be exceeded to meet min_per_speaker).
        min_per_speaker (int): Minimum number of target trials to generate per speaker (ignores max_usage if needed).

    Returns:
        List[Tuple[str, str, str]]: List of (path1, path2, "target") trials.
    """
    trials = []
    speakers = list(bonafide_samples.keys())

    # Step 1: Satisfy minimum target trials per speaker (even if overusing samples)
    for speaker in speakers:
        samples = bonafide_samples[speaker]
        if len(samples) < 2:
            continue  # Not enough samples for pairing
        for _ in range(min_per_speaker):
            s1, s2 = random.sample(samples, 2)
            trials.append((s1, s2, "target"))
            usage_counter[s1] += 1
            usage_counter[s2] += 1

    remaining = n - len(trials)
    print(f"Generated {len(trials)} forced target trials. Now generating {remaining} remaining trials.")

    # Step 2: Fill remaining trials, respecting max_usage
    with tqdm(total=remaining, desc="Creating target trials") as pbar:
        while len(trials) < n:
            eligible = [
                s for s in speakers if sum(usage_counter[p] < max_usage for p in bonafide_samples[s]) >= 2
            ]
            if not eligible:
                break
            speaker = random.choice(eligible)
            samples = bonafide_samples[speaker]
            s1 = select_random_least_used(samples, usage_counter)
            s2 = select_random_least_used([s for s in samples if s != s1], usage_counter)
            trials.append((s1, s2, "target"))
            usage_counter[s1] += 1
            usage_counter[s2] += 1
            pbar.update(1)

    return trials


def create_nontarget_trials(bonafide_samples, n, usage_counter, max_usage):
    """
    Generate trials where both audio samples are bonafide but from different speakers.

    Args:
        bonafide_samples (dict): Bonafide samples grouped by speaker.
        n (int): Number of trials to generate.
        usage_counter (Counter): Tracks usage of each file.
        max_usage (int): Maximum allowed usage for any one sample.

    Returns:
        List[Tuple[str, str, str]]: List of (path1, path2, "nontarget") trials.
    """
    trials = []
    speakers = list(bonafide_samples.keys())

    with tqdm(total=n, desc="Creating nontarget trials") as pbar:
        while len(trials) < n:
            eligible = [
                s for s in speakers if any(usage_counter[p] < max_usage for p in bonafide_samples[s])
            ]
            if len(eligible) < 2:
                break

            spk1, spk2 = random.sample(eligible, 2)
            s1 = select_random_least_used(bonafide_samples[spk1], usage_counter)
            s2 = select_random_least_used(bonafide_samples[spk2], usage_counter)

            trials.append((s1, s2, "nontarget"))
            usage_counter[s1] += 1
            usage_counter[s2] += 1
            pbar.update(1)

    return trials


def create_spoof_trials(bonafide_samples, spoofed_samples, n, usage_counter, max_usage, min_per_speaker=0):
    """
    Generate spoof trials: the first audio is bonafide, the second is spoofed,
    both from the same speaker.

    Ensures a minimum number of spoof trials per speaker are created, ignoring
    the max_usage limit when necessary.

    Args:
        bonafide_samples (dict): {speaker_id: list of bonafide file paths}
        spoofed_samples (dict): {speaker_id: list of spoofed file paths}
        n (int): Total number of spoof trials to generate.
        usage_counter (Counter): Tracks how often each sample has been used.
        max_usage (int): Maximum allowed usage for any sample (can be exceeded to meet min_per_speaker).
        min_per_speaker (int): Minimum number of spoof trials to generate per speaker (ignores max_usage if needed).

    Returns:
        List[Tuple[str, str, str]]: List of (path1, path2, "spoof") trials.
    """
    trials = []
    speakers = list(set(bonafide_samples.keys()) & set(spoofed_samples.keys()))

    # Step 1: Satisfy minimum spoof trials per speaker (even if overusing samples)
    for speaker in speakers:
        bonafide = bonafide_samples[speaker]
        spoofed = spoofed_samples[speaker]
        if not bonafide or not spoofed:
            continue
        for _ in range(min_per_speaker):
            s1 = random.choice(bonafide)
            s2 = random.choice(spoofed)
            trials.append((s1, s2, "spoof"))
            usage_counter[s1] += 1
            usage_counter[s2] += 1

    remaining = n - len(trials)
    print(f"Generated {len(trials)} forced spoof trials. Now generating {remaining} remaining trials.")

    # Step 2: Fill remaining trials, respecting max_usage
    with tqdm(total=remaining, desc="Creating spoof trials") as pbar:
        while len(trials) < n:
            eligible = [
                s for s in speakers if
                any(usage_counter[p] < max_usage for p in bonafide_samples[s]) and
                any(usage_counter[p] < max_usage for p in spoofed_samples[s])
            ]
            if not eligible:
                break
            speaker = random.choice(eligible)
            s1 = select_random_least_used(bonafide_samples[speaker], usage_counter)
            s2 = select_random_least_used(spoofed_samples[speaker], usage_counter)
            trials.append((s1, s2, "spoof"))
            usage_counter[s1] += 1
            usage_counter[s2] += 1
            pbar.update(1)

    return trials


def main(input_file, output_file, num_target, num_nontarget, num_spoof, max_usage, seed=42, min_target_per_speaker=0, min_spoof_per_speaker=0):
    """
    Main function to load data, generate trials, and write to output file.

    Args:
        input_file (str): Input .txt file with path,speaker,label lines.
        output_file (str): Output file to save generated trials.
        num_target (int): Number of target (same speaker, bonafide-bonafide) trials.
        num_nontarget (int): Number of nontarget (diff speaker, bonafide-bonafide) trials.
        num_spoof (int): Number of spoof (same speaker, bonafide-spoof) trials.
        max_usage (int): Maximum number of times each sample can be used.
        seed (int): Random seed for reproducibility.
        min_target_per_speaker (int): Minimum number of target trials to generate per speaker (ignores max_usage if needed).
        min_spoof_per_speaker (int): Minimum number of spoof trials to generate per speaker (ignores max_usage if needed).
    """
    random.seed(seed)
    usage_counter = Counter()

    bonafide_samples, spoofed_samples = load_samples(input_file)

    target_trials = create_target_trials(bonafide_samples, num_target, usage_counter, max_usage, min_target_per_speaker)
    spoof_trials = create_spoof_trials(bonafide_samples, spoofed_samples, num_spoof, usage_counter, max_usage, min_spoof_per_speaker)
    nontarget_trials = create_nontarget_trials(bonafide_samples, num_nontarget, usage_counter, max_usage)

    total_trials = target_trials + nontarget_trials + spoof_trials

    with open(output_file, 'w') as f:
        for path1, path2, label in total_trials:
            f.write(f"{path1},{path2},{label}\n")

    print(f" Generated {len(total_trials)} trials: "
          f"{len(target_trials)} target, {len(nontarget_trials)} nontarget, {len(spoof_trials)} spoof.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate speaker verification trials")
    parser.add_argument("input_file", type=str, help="Input .txt file with path,speaker,label format")
    parser.add_argument("output_file", type=str, help="Output file for generated trials")
    parser.add_argument("--target", type=int, default=15000, help="Number of target trials")
    parser.add_argument("--nontarget", type=int, default=15000, help="Number of nontarget trials")
    parser.add_argument("--spoof", type=int, default=15000, help="Number of spoof trials")
    parser.add_argument("--max_usage", type=int, default=5, help="Max usage per sample")
    parser.add_argument("--min_target_per_speaker", type=int, default=10, help="Minimum number of target trials per speaker")
    parser.add_argument("--min_spoof_per_speaker", type=int, default=10, help="Minimum number of spoof trials per speaker")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.target, args.nontarget, args.spoof, args.max_usage, args.seed, args.min_target_per_speaker, args.min_spoof_per_speaker)
