# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import random
from pathlib import Path
import torch
import torchaudio
import logging

logger = logging.getLogger(__name__)


class MUSANAugmentation:
    """
    Adds additive noise from MUSAN dataset to the waveform with a randomly chosen
    noise type and signal-to-noise ratio (SNR).

    Args:
        config (dict):
            - 'folder' (str or Path): Path to MUSAN folder with noise subfolders.
            - 'noise_types' (list of str): List of noise categories to use (e.g., ['noise', 'music']).
            - 'snr_range' (list or dict): Either [min_snr, max_snr] or dict with noise_type keys and [min,max].
            - 'prob' (float): Probability of applying MUSAN augmentation (default 0.0).
    """

    def __init__(self, config):
        self.prob = config.get('prob', 0.0)
        self.noise_types = config.get('noise_types', [])
        self.snr_range = config.get('snr_range', [5, 20])

        self.musan_folder = config.get('folder', None)
        self.musan_files = {}

        if self.musan_folder:
            musan_path = Path(self.musan_folder)
            for ntype in self.noise_types:
                noise_path = musan_path / ntype
                self.musan_files[ntype] = list(noise_path.glob("**/*.wav"))
                if not self.musan_files[ntype]:
                    logger.warning(f"No MUSAN files found for noise type '{ntype}' in {noise_path}")
        
        logger.info(f"Initialised MUSANAugmentation with parameters:\n{self}")


    def __str__(self):
        lens = {k: len(v) for k,v in self.musan_files.items()}
        return (
            f"MUSANAugmentation(probability={self.prob}, "
            f"musan_folder={self.musan_folder}, "
            f"noise_types={self.noise_types}, "
            f"snr_range={self.snr_range}, "
            f"files={lens})"
        )


    def __call__(self, wav):
        """
        Adds MUSAN noise augmentation to the input waveform.

        Args:
            wav (torch.Tensor): 1D tensor of the audio waveform.

        Returns:
            torch.Tensor: 1D tensor of the noise augmented waveform.
        """
        if not self.musan_files:
            logger.warning("MUSAN files list empty, skipping MUSAN augmentation")
            return wav

        ntype = random.choice(list(self.musan_files.keys()))
        noise_list = self.musan_files.get(ntype, [])
        if not noise_list:
            logger.warning(f"No noise files available for MUSAN noise type '{ntype}'")
            return wav

        noise_path = random.choice(noise_list)
        noise_wav, _ = torchaudio.load(str(noise_path))
        if noise_wav.dim() == 2:
            noise_wav = noise_wav.mean(dim=0)
        noise_wav = noise_wav.squeeze()

        if isinstance(self.snr_range, dict):
            snr_min, snr_max = self.snr_range.get(ntype, [5, 20])
        else:
            snr_min, snr_max = self.snr_range

        snr_db = random.uniform(snr_min, snr_max)

        if noise_wav.size(0) < wav.size(0):
            repeat_factor = wav.size(0) // noise_wav.size(0) + 1
            noise_wav = noise_wav.repeat(repeat_factor)[:wav.size(0)] if noise_wav.dim() == 1 else noise_wav
        else:
            start = random.randint(0, noise_wav.size(0) - wav.size(0))
            noise_wav = noise_wav[start:start + wav.size(0)]

        wav_power = wav.norm(p=2)
        noise_power = noise_wav.norm(p=2)
        factor = wav_power / (10 ** (snr_db / 20)) / (noise_power + 1e-10)
        wav = wav + factor * noise_wav
        return wav
