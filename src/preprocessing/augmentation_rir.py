# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import random
from pathlib import Path
import torch
import torchaudio
import scipy.signal
import logging

logger = logging.getLogger(__name__)


class RIRAugmentation:
    """
    Applies Room Impulse Response (RIR) augmentation by convolving the input waveform
    with a randomly selected RIR waveform.

    Args:
        config (dict):
            - 'folder' (str or Path): Path to directory containing RIR wav files (default None).
            - 'prob' (float): Probability of applying RIR augmentation (default 0.0).
    """

    def __init__(self, config):
        self.prob = config.get('prob', 0.0)
        self.rir_folder = config.get('folder', None)
        self.rir_files = []

        if self.rir_folder:
            rir_path = Path(self.rir_folder)
            self.rir_files = list(rir_path.glob("**/*.wav"))
            if not self.rir_files:
                logger.warning(f"No RIR files found in {self.rir_folder}")
        
        logger.info(f"Initialised RIRAugmentation with parameters:\n{self}")
    

    def __str__(self):
        return (
            f"RIRAugmentation(probability={self.prob}, "
            f"rir_folder={self.rir_folder}, "
            f"rir_files={len(self.rir_files)})"
        )
    

    def __call__(self, wav):
        """
        Applies the RIR augmentation on the input waveform.

        Args:
            wav (torch.Tensor): 1D tensor representing the audio waveform.

        Returns:
            torch.Tensor: 1D tensor of the augmented waveform after convolution with a randomly selected RIR.
        """
        if not self.rir_files:
            logger.warning("RIR files list empty, skipping RIR augmentation")
            return wav

        for _ in range(10):  # Try up to 10 different RIRs
            rir_path = random.choice(self.rir_files)
            try:
                rir_wav, _ = torchaudio.load(str(rir_path))

                # Convert to mono if necessary
                if rir_wav.shape[0] > 1:
                    rir_wav = torch.mean(rir_wav, dim=0, keepdim=True)

                rir_wav = rir_wav / torch.norm(rir_wav, p=2)

                wav = self.fft_convolve(wav.squeeze(), rir_wav.squeeze())

                return wav

            except Exception as e:
                logger.warning(f"Failed to load or apply RIR from {rir_path}: {e}")
                continue

        logger.warning("All RIR augmentation attempts failed. Returning original wav.")
        return wav
    

    def fft_convolve(self, wav, rir):
        """
        Applies convolution of a waveform with a RIR 
        using FFT-based convolution for improved performance over direct convolution.

        Args:
            wav (torch.Tensor): 1D tensor representing the audio waveform.
            rir (torch.Tensor): 1D tensor representing the RIR.

        Returns:
            torch.Tensor: 1D tensor containing the convolved audio waveform.
        """
        wav_np = wav.numpy()
        rir_np = rir.numpy()

        # FFT convolution and truncate to original length
        convolved = scipy.signal.fftconvolve(wav_np, rir_np, mode="full")[:len(wav_np)]
        return torch.tensor(convolved, dtype=wav.dtype)
