# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import re
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class TrialDataset(Dataset):
    def __init__(self, wav_loader, trial_file, base_audio_path, batch_size=1, num_workers=0, pin_memory=True, prefetch_factor=None):
        """
        A PyTorch Dataset for loading audio data specified in a trial file.

        Each trial line is expected to contain at least two paths, from which unique audio file paths are extracted.
        The trial file is structured as follows and the components can be split by whitespaces and/or commas:
        path1, path2, label
        The dataset loads and returns audio features using a provided wav_loader callable.

        Attributes:
            wav_loader (WavLoader): Function that loads and processes an audio file given its path.
            trial_file (str): Path to the trial file with lines path1, path2, label.
            base_audio_path (str): Base directory path for resolving audio file paths.
            batch_size (int): Batch size for the DataLoader (default: 1).
            num_workers (int): Number of subprocesses to use for data loading (default: 0).
            pin_memory (bool): If True, DataLoader will copy Tensors into CUDA pinned memory (default: True).
            prefetch_factor (int): Number of batches to prefetch per worker (default: None).
        """
        self.wav_loader = wav_loader
        self.trial_file = trial_file
        self.paths, self.trials = self._load_unique_paths(self.trial_file)
        self.base_audio_path = base_audio_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

        with open(self.trial_file, 'r') as file:
            line_count = sum(1 for line in file)

        logger.info(f"Initialised dataset with {line_count} trails, {len(self.paths)} unique paths and parameters:\n{self}")
    
    def __str__(self):
        return(
            f"TrialDataset(trial_file={self.trial_file}, "
            f"base_audio_path={self.base_audio_path}, "
            f"batch_size={self.batch_size}, "
            f"num_workers={self.num_workers}, "
            f"pin_memory={self.pin_memory}, "
            f"prefetch_factor={self.prefetch_factor})"
        )

    def __len__(self):
        """Returns the total number of unique audio file paths."""
        return len(self.paths)

    def __getitem__(self, index):
        """
        Loads and returns features for the audio file at the given index.

        Args:
            index (int): Index of the audio path in the dataset.

        Returns:
            Tensor: Output of the wav_loader for the audio file.
        """
        path = self.paths[index]
        path = Path(self.base_audio_path) / path
        features = self.wav_loader(path)
        return features
    
    def _load_unique_paths(self, trial_file):
        """
        Parses the trial file and extracts unique audio paths.

        Args:
            trial_file (str): Path to the trial file with lines path1, path2, label.

        Returns:
            Tuple[List[str], List[List[str]]]: A tuple of (unique_paths, trial_entries).
        """
        paths_set = set()
        trial_entries = []

        with open(trial_file, 'r') as f:
            for line in tqdm(f, desc="Reading Trials"):
                trial = re.split(r'[,\s]+', line.strip())
                path1, path2, label = *map(Path, trial[:2]), trial[2].lower()
                paths_set.update([path1, path2])
                trial_entries.append([path1, path2, label])

        unique_paths = sorted(list(paths_set))
        return unique_paths, trial_entries

    
    def get_dataloader(self):
        """
        Returns a PyTorch DataLoader for the dataset.

        Returns:
            DataLoader: Configured DataLoader instance.
        """
        def collate_fn(batch):  # when using this, unpack audio, length in evaluation.py#163
            wavs_len = torch.tensor([wav.shape[0] for wav in batch], dtype=torch.long)
            return pad_sequence(batch, batch_first=True), wavs_len
        return DataLoader(self, batch_size=1, shuffle=False, num_workers=0, pin_memory=self.pin_memory, drop_last=False)