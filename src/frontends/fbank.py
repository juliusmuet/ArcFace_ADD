# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torch.nn as nn
from torchaudio.compliance import kaldi as Kaldi
import logging

logger = logging.getLogger(__name__)


class FBankFrontend(nn.Module):
    def __init__(self, config={}):
        """
        A frontend module for extracting Mel-filterbank (FBank) features from raw audio waveforms
        using Kaldi's FBank feature extraction implementation.
        FBank features need to be calculated when single waveform is loaded!

        Args:
            config (dict): A configuration dictionary containing the following optional keys:
                - 'n_mels' (int): Number of mel bins to use (default: 80)
                - 'dither' (float): Dithering to add to audio (default: 0)
                - 'mean_norm' (bool): Whether to apply mean normalisation to the features (default: True)
                - 'sample_rate' (int): Sample rate of the input waveform (default: 16000)
        """
        super().__init__()
        self.sample_rate = config.get('sample_rate', 16000)
        self.n_mels = config.get('n_mels', 80)
        self.dither = config.get('dither', 0)
        self.mean_norm = config.get('mean_norm', True)
        logger.info(f"Initialised FBank Frontend with parameters:\n{self}")
    

    def __str__(self):
        return (
            f"FBankFrontend(sample_rate={self.sample_rate}, "
            f"n_mels={self.n_mels}, "
            f"dither={self.dither}, "
            f"mean_norm={self.mean_norm})"
        )


    def forward(self, wav):
        """
        Compute Mel-filterbank features from the input waveform.

        Args:
            wavs (torch.Tensor): A tensor of shape (1, seq_len) or (seq_len,) representing raw audio signals.
                                 If 1D, it is reshaped to (1, seq_len).

        Returns:
            torch.Tensor: A 2D tensor of shape (num_frames, n_mels) containing FBank features.
        """
        # Reshape (seq_len,) to (1, seq_len)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        
        if wav.dim() != 2:
            raise ValueError(f"Expected shape of wavs to be (seq_len,) or (1, seq_len), but got shape {wav.shape}")

        # Calculate features
        features = Kaldi.fbank(wav, num_mel_bins=self.n_mels, sample_frequency=self.sample_rate, dither=self.dither)
        
        # Cepstral Mean Normalisation (CMN) on filterbanks
        if self.mean_norm:
            features = features - features.mean(0, keepdim=True)
        
        return features # Shape (num_frames, n_mels)