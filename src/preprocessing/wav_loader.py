# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import random
import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import numpy as np
import logging
from frontends.fbank import FBankFrontend
from preprocessing.augmentation_rir import RIRAugmentation
from preprocessing.augmentation_musan import MUSANAugmentation
from preprocessing.augmentation_rawboost import RawBoostAugmentation

logger = logging.getLogger(__name__)


class WavLoader():
    """
    A preprocessing class for loading and processing audio waveforms from file paths,
    with optional augmentation (RIR, MUSAN, RawBoost).

    Args:
        config (dict):
            A dictionary containing configuration values containing the following optional keys:
                - 'duration' (float): Desired duration (in seconds) of the waveform (default: -1 / no resizing).
                - 'sample_rate' (int): Target sample rate for audio files (default: 16000 Hz).
                - 'fbank_config' (dict): Configuration for the FBank frontend (default: None).
                - 'remove_leading_trailing_silence' (bool): Whether to remove leading and trailing silence (default: True).
                - 'remove_all_silence' (bool): Whether to remove all silence (default: False).
                - 'silence_threshold' (int): Threshold db at which audio is considered as silence (default: 30).
                - 'augmentations' (dict): Configuration for augmentations, passed to augmentation classes.
    """

    def __init__(self, config={}):
        self.duration = config.get('duration', -1)
        self.sample_rate = config.get('sample_rate', 16000)
        self.silence_treshold = config.get('silence_threshold', 30)
        self.remove_leading_trailing_silence = config.get('remove_leading_trailing_silence', True)
        self.remove_all_silence = config.get('remove_all_silence', False)
        fbank_config = config.get('fbank_config', None)
        self.fbank = None
        if fbank_config is not None:
            self.fbank = FBankFrontend(fbank_config)
        
        # Initialize augmentation classes
        aug_cfg = config.get('augmentations', {})
        self.rir_aug = RIRAugmentation(aug_cfg.get('rir', {})) if 'rir' in aug_cfg else None
        self.musan_aug = MUSANAugmentation(aug_cfg.get('musan', {})) if 'musan' in aug_cfg else None
        rawboost_cfg = aug_cfg.get('rawboost', {})
        rawboost_cfg['sample_rate'] = self.sample_rate
        self.rawboost_aug = RawBoostAugmentation(rawboost_cfg) if 'rawboost' in aug_cfg else None
        
        logger.info(f"Initialised WavLoader with parameters:\n{self}")


    def __str__(self):
        return (
            f"WavLoader(duration={self.duration}, "
            f"sample_rate={self.sample_rate}, "
            f"silence_threshold={self.silence_treshold}, "
            f"remove_leading_trailing_silence={self.remove_leading_trailing_silence}, "
            f"remove_all_silence={self.remove_all_silence}, "
            f"fbank_config={'given' if self.fbank else 'None'})"
        )


    def __call__(self, wav_path, is_vocoder_aug=False):
        """
        Loads and processes a waveform from a given file path.

        Args:
            wav_path (str): Path to the audio file.
            is_vocoder_aug (bool): Flag for vocoder augmentation data (default: False).

        Returns:
            torch.Tensor: A 1D tensor representing the processed mono audio waveform
                          or 2D tensor representing the FBank features from the processed waveform.
        """
        wav = self._load_waveform(wav_path) # Load waveform

        # Remove silence parts
        if self.remove_leading_trailing_silence:
            wav = self._remove_leading_trailing_silence(wav, self.silence_treshold)
            if wav.numel() == 0:
                logger.warning(f"The audio {wav_path} contains only silence (leading/trailing).")
        if self.remove_all_silence:
            wav = self._remove_all_silence(wav, self.silence_treshold)
            if wav.numel() == 0:
                logger.warning(f"The audio {wav_path} contains only silence (entire audio).")

        # Slice / pad waveform if duration is not equal to -1
        if self.duration != -1.0:
            wav = self._resize_waveform(wav)    

        # Apply augmentations only if not vocoder-aug data
        if not is_vocoder_aug:
            wav = self._apply_augmentations(wav)

        # Compute FBank features if FBankFrontend is given
        if self.fbank is not None:
            with torch.no_grad():
                wav = self.fbank(wav)

        return wav
    

    def _load_waveform(self, wav_path):
        """
        Loads a waveform from a file and converts it to mono and target sample rate if necessary.

        Args:
            wav_path (str): Path to the audio file.

        Returns:
            torch.Tensor: A 1D tensor representing the loaded and preprocessed waveform.
        """
        # Load waveform from path
        wav, sr = torchaudio.load(wav_path) # Shape (channels, seq_len)
        
        # Resample audio if needed
        if sr != self.sample_rate:
            self.resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            wav = self.resampler(wav)

        # Channel averaging to convert stereo or multi-channel audio to mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True) # Shape (1, seq_len)

        return wav.squeeze(0)    # Shape (seq_len,)
    

    def _resize_waveform(self, wav):
        """
        Adjusts waveform to a fixed length by slicing or padding by repeating the input.

        Args:
            wav (torch.Tensor): A 1D tensor representing the input waveform.

        Returns:
            torch.Tensor: A 1D tensor of the waveform with the desired fixed length.
        """
        chunk_len = int(self.duration * self.sample_rate)
        wav_len = wav.shape[0]
        if wav_len >= chunk_len:   # Slicing by selecting random part of waveform
            chunk_start = random.randint(0, wav_len - chunk_len)
            wav = wav[chunk_start:chunk_start + chunk_len]
            # re-clone the data to avoid memory leakage
            if type(wav) == torch.Tensor:
                wav = wav.clone()
            else:  # np.array
                wav = wav.copy()
        else:   # Padding by repeating the input
            repeat_factor = chunk_len // wav_len + 1
            repeat_shape = repeat_factor if len(wav.shape) == 1 else (repeat_factor, 1)    # if input may be 2D instead of 1D
            if type(wav) == torch.Tensor:
                wav = wav.repeat(repeat_shape)
            else:  # np.array
                wav = np.tile(wav, repeat_shape)
            wav = wav[:chunk_len]

        return wav
    

    def _remove_leading_trailing_silence(self, wav, silence_threshold):
        """
        Removes leading and trailing silence from a mono waveform using Silero VAD.

        Args:
            wav (torch.Tensor): A 1D tensor representing the input waveform.
            silence_threshold (int): Threshold db at which audio is considered as silence.


        Returns:
            torch.Tensor: A 1D waveform tensor with leading and trailing silence removed.
                          Returns an empty tensor if no speech is detected.
        """
        # Convert to numpy
        wav_np = wav.numpy()

        # Use librosa to trim silence
        trimmed_wav, _ = librosa.effects.trim(wav_np, top_db=silence_threshold)

        # Check if output is empty (i.e., all silence)
        if trimmed_wav.size == 0:
            return torch.empty(0)

        # Convert back to torch
        return torch.from_numpy(trimmed_wav)
    

    def _remove_all_silence(self, wav, silence_threshold):
        """
        Removes all silent parts from a mono waveform using Silero VAD.

        Args:
            wav (torch.Tensor): A 1D tensor representing the input waveform.
            silence_threshold (int): Threshold db at which audio is considered as silence.

        Returns:
            torch.Tensor: A waveform tensor with all silence removed.
                          Returns an empty tensor if no speech is detected.
        """
        # Convert to numpy
        wav_np = wav.numpy()

        # Use librosa to get non-silent intervals
        intervals = librosa.effects.split(wav_np, top_db=silence_threshold)

        # Check if no non-silent intervals were found
        if len(intervals) == 0:
            return torch.empty(0)

        # Stitch together non-silent intervals
        non_silent = np.concatenate([wav_np[start:end] for start, end in intervals], axis=0)

        # Convert back to torch
        return torch.from_numpy(non_silent)
    

    def _apply_augmentations(self, wav):
        if self.rir_aug and random.random() < self.rir_aug.prob:
            wav = self.rir_aug(wav)

        if self.musan_aug and random.random() < self.musan_aug.prob:
            wav = self.musan_aug(wav)

        if self.rawboost_aug and random.random() < self.rawboost_aug.prob:
            wav = self.rawboost_aug(wav)

        return wav
