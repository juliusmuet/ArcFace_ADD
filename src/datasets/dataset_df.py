# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
from utils.model_getter import get_speaker_label_encoder, get_batch_sampler

logger = logging.getLogger(__name__)


def load_data(dataset_string, dataset_string_genuine_only, filter_out_vocoder, vocoder_as_genuine):
    """
    Function to load all datapoints for specified datasets.
    Args:
        dataset_string (str): A string of the deepfake-datasets to be used (separated by spaces).
        dataset_string_genuine_only (str): Same as asv_datasets but only genuine data is used from
                                           those datasets (i.e. no audio deepfakes)
        filter_out_vocoder (bool): A flag to filter out all "vocoder only" audios
        vocoder_as_genuine (bool): A flag to change the label of "vocoder only" audios to "genuine",
                                   i.e. is_spoof is False instead of True for those audios.


    Returns
        list[dict]: A list of dict-elements. Each element represents one audio-file and contains (at least) the
                    following attributes:
                    speaker_id: The speaker ID
                    is_spoof: Boolean flag if spoof or not
                    wav_path: The path to the audio (can also be mp3 or flac, not just wav)
    """
    pass


class ASVDataset_DF(Dataset):
    """
    A PyTorch Dataset for Automatic Speaker Verification (ASV) with deepfakes.
    
    Args:
        dataset_string (str): A string of the deepfake-datasets to be used (separated by spaces).
        dataset_string_genuine_only (str): Same as asv_datasets but only genuine data is used from
                                           those datasets (i.e. no audio deepfakes)
        wav_loader (WavLoader): A function to load audio files and return the waveform or its features.
        speaker_label_encoder (str): Name of the label encoder class that can be instantiated with a speaker list.
        filter_out_vocoder (bool): A flag to filter out all "vocoder only" audios (default: False).
        vocoder_as_genuine (bool): A flag to change the label of "vocoder only" audios to "genuine",
                                   i.e. is_spoof is False instead of True for those audios (default: True).
    """

    def __init__(self, dataset_string, dataset_string_genuine_only, wav_loader, speaker_label_encoder, filter_out_vocoder=False, vocoder_as_genuine=True):
        self.dataset_string = dataset_string
        self.dataset_string_genuine_only = dataset_string_genuine_only
        self.filter_out_vocoder = filter_out_vocoder
        self.vocoder_as_genuine = vocoder_as_genuine

        self.data_points = load_data(dataset_string, dataset_string_genuine_only, filter_out_vocoder, vocoder_as_genuine)
        self.wav_loader = wav_loader
        self.label_encoder = get_speaker_label_encoder(speaker_label_encoder)(*self.get_speakers_by_genuine_class())

        logger.info(f"Initialised dataset with {len(self.data_points)} datapoints and parameters:\n{self}")


    def __str__(self):
        return(
            f"ASVDataset_DF(dataset_string={self.dataset_string}, "
            f"dataset_string_genuine_only={self.dataset_string_genuine_only}, "
            f"encoder={self.label_encoder.__class__.__name__}, "
            f"filter_out_vocoder={self.filter_out_vocoder}, "
            f"vocoder_as_genuine={self.vocoder_as_genuine})"
        )


    def __len__(self):
        """
        Returns:
            int: Total number of data points.
        """
        return len(self.data_points)

 
    def __getitem__(self, index):
        """
        Retrieves a single data sample.
        
        Args:
            index (int): Index of the sample to retrieve.
        
        Returns:
            tuple: (features, speaker_id (encoded), genuine_label)
                - features: Extracted features from the audio file.
                - speaker_id: Encoded speaker label.
                - genuine_label: 1 for genuine, 0 for spoof.
        """
        data = self.data_points[index]
        
        wav_path = Path(data['wav_path'])
        is_vocoded = (data.get('is_spoof', False) == False) and data.get('method_type', '') and (data.get('method_type', '').lower() == 'vocoder')
        features = self.wav_loader(wav_path, is_vocoded)

        genuine = 0 if data.get('is_spoof', False) else 1

        spk = data['speaker_id']
        spkid = self.label_encoder(spk, genuine)

        return features, spkid, genuine


    def get_speaker_list(self):
        """
        Returns:
            list[str]: Unique speaker IDs regardless of genuine or deepfake.
        """
        speakers = list(set([x["speaker_id"] for x in tqdm(self.data_points, desc="Getting speaker list")]))
        return speakers


    def get_speaker_count(self):
        """
        Returns:
            int: Number of unique speakers regardless of genuine or deepfake.
        """
        return len(self.get_speaker_list())


    def get_speakers_by_genuine_class(self):
        """
        Returns:
            tuple (list[str], list[str]):
            Unique speaker IDs with at least one genuine sample.
            Unique speaker IDs with at least one deepfake sample.
        """
        genuine_speakers = set()
        deepfake_speakers = set()
        for data in tqdm(self.data_points, desc="Getting speaker list by genuine class"):
            if not data.get('is_spoof', False):
                genuine_speakers.add(data['speaker_id'])
            else:
                deepfake_speakers.add(data['speaker_id'])
        return list(genuine_speakers), list(deepfake_speakers)
    

    def get_num_classes(self):
        """
        Returns:
            int: Number of classes created by the label encoder.
        """
        return len(self.label_encoder.lab2ind.keys())


    def get_speaker_label_for_idx(self, index):
        """
        Get the encoded speaker label for a given index.
        
        Args:
            index (int): Index of the data point.
        
        Returns:
            int: Encoded speaker ID.
        """
        spk = self.data_points[index]["speaker_id"]
        genuine = self.get_genuine_label_for_idx(index)
        spkid = self.label_encoder(spk, genuine)
        return spkid


    def get_genuine_label_for_idx(self, index):
        """
        Get the genuine/spoof label for a given index.
        
        Args:
            index (int): Index of the data point.
        
        Returns:
            int: 1 if genuine, 0 if spoof.
        """
        return 0 if self.data_points[index].get('is_spoof', False) else 1

  
    def get_dataloader(self, config_dataloader, sampler_config=None):
        """
        Returns the appropriate DataLoader based on the specified sampler.

        Args:
            config_dataloader (dict): Configuration parameters for the DataLoader.
                                      Should include keys and values corresponding to the arguments 
                                      expected by either `_get_standard_dataloader` or `_get_balanced_dataloader`.
            sampler_config (dict): Config of sampler to use (default: None).

        Returns:
            DataLoader: A PyTorch DataLoader configured based on the sampling strategy.
        """
        if sampler_config != None:
            return self._get_balanced_dataloader(sampler_config=sampler_config, **config_dataloader)
        else:
            return self._get_standard_dataloader(**config_dataloader)


    def _get_standard_dataloader(self, batch_size=8, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=4, drop_last=True):
        """
        Return a DataLoader with no custom sampling.
        
        Args:
            batch_size (int): Batch size for the DataLoader (default: 8).
            shuffle (bool): Whether to shuffle the data (default: True).
            num_workers (int): Number of worker processes for data loading (default: 8).
            pin_memory (bool): Whether to pin memory for faster data transfer (default: True).
            prefetch_factor (int): Number of batches to prefetch per worker (default: 4).
            drop_last (bool): Whether to drop the last batch if it's smaller than batch_size (default: True).
                              Should be false for evaluation / inference.
        
        Returns:
            DataLoader: PyTorch DataLoader for the dataset.
        """
        logger.info("Initialised Dataloader with no custom sampling with parameters:\n"
            f"batch_size={batch_size}, "
            f"shuffle={shuffle}, "
            f"num_workers={num_workers}, "
            f"pin_memory={pin_memory}, "
            f"prefetch_factor={prefetch_factor}, "
            f"drop_last={drop_last}"
        )
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor, drop_last=drop_last)

   
    def _get_balanced_dataloader(self, sampler_config, num_workers=8, pin_memory=True, prefetch_factor=4):
        """
        Returns a DataLoader with speaker-balanced sampling.

        Args:
            sampler_config (dict): Config of the sampler to use.
            n_speakers_per_batch (int): Number of unique speakers per batch.
            n_utterances_per_speaker (int): Number of utterances per speaker per batch.
            num_workers (int): Number of worker processes for data loading (default: 8).
            pin_memory (bool): Whether to pin memory for faster data transfer (default: True).
            prefetch_factor (int): Number of batches to prefetch per worker (default: 4).

        Returns:
            DataLoader: PyTorch DataLoader with batches balanced by speaker.
        """
        sampler_name = sampler_config['sampler']
        sampler_config.pop('sampler')
        
        logger.info("Initialised Dataloader with speaker-balanced sampling with parameters:\n"
            f"batch_sampler={sampler_name}, "
            f"num_workers={num_workers}, "
            f"pin_memory={pin_memory}, "
            f"prefetch_factor={prefetch_factor}"
        )
        
        sampler = get_batch_sampler(sampler_name)(self, **sampler_config)
        sampler_config['sampler'] = sampler_name
        return DataLoader(self, batch_sampler=sampler, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
