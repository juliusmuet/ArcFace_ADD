# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from collections import defaultdict
import random
from torch.utils.data import Sampler
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class SpeakerBalancedBatchSampler(Sampler):
    """
    A PyTorch Sampler that yields batches containing a fixed number of speakers,
    and a fixed number of utterances per speaker to ensure speaker-balanced sampling.

    Args:
        dataset (Dataset): The dataset containing data points with 'speaker_id' in each item.
        n_speakers_per_batch (int): Number of unique speakers per batch.
        n_utterances_per_speaker (int): Number of utterances to sample per speaker in each batch.

    Yields:
        List[int]: A list of indices representing one batch containing balanced speaker samples.
    """

    def __init__(self, dataset, n_speakers_per_batch, n_utterances_per_speaker):
        self.n_speakers_per_batch = n_speakers_per_batch
        self.n_utterances_per_speaker = n_utterances_per_speaker

        # Calculate number of batches per epoch
        self.batch_size = self.n_speakers_per_batch * self.n_utterances_per_speaker
        self.num_samples = len(dataset)
        self.num_batches = self.num_samples // self.batch_size

        # Build index mapping: speaker_id -> list of indices
        self.speaker_to_indices = defaultdict(list)
        for idx, data_point in tqdm(enumerate(dataset.data_points), desc="Initialising SpeakerBalancedBatchSampler"):
            self.speaker_to_indices[data_point['speaker_id']].append(idx)
        self.speaker_ids = list(self.speaker_to_indices.keys())   # List of unique speakers

        logger.info(f"Initialised SpeakerBalancedBatchSampler with parameters:\n{self}")


    def __str__(self):
        return (f"Total samples: {self.num_samples}, "
                f"Number of speakers: {len(self.speaker_ids)}, "
                f"Speakers per batch: {self.n_speakers_per_batch}, "
                f"Utterances per speaker: {self.n_utterances_per_speaker}, "
                f"Batch size: {self.batch_size}, "
                f"Total batches per epoch: {self.num_batches}")


    def __iter__(self):
        """
        Yields batches of indices, each batch containing:
        - n_speakers_per_batch unique speakers
        - n_utterances_per_speaker utterances per speaker
        """
        for _ in range(self.num_batches):
            # Sample speakers for this batch
            if len(self.speaker_ids) >= self.n_speakers_per_batch:
                selected_speakers = random.sample(self.speaker_ids, self.n_speakers_per_batch)
            else:
                selected_speakers = random.choices(self.speaker_ids, k=self.n_speakers_per_batch)
                logger.warning("Sampling speakers with replacement to complete batch.")
            
            batch_indices = []
            
            # For each speaker, sample utterances
            for spk in selected_speakers:
                indices = self.speaker_to_indices[spk]
                
                if len(indices) >= self.n_utterances_per_speaker:
                    selected_indices = random.sample(indices, self.n_utterances_per_speaker)
                # If speaker has fewer utterances than needed, sample with replacement
                else:
                    selected_indices = random.choices(indices, k=self.n_utterances_per_speaker)
                    logger.warning(f"Speaker {spk} has insufficient samples. Sampling with replacement.")
                
                batch_indices.extend(selected_indices)
            
            yield batch_indices


    def __len__(self):
        """Returns the number of batches per epoch."""
        return self.num_batches
    


class SpeakerGenuineFakeBalancedSampler(Sampler):
    """
    A PyTorch Sampler that yields batches containing a fixed number of speakers,
    with a fixed number of genuine and deepfake utterances per speaker.

    Args:
        dataset (Dataset): The dataset containing 'speaker_id' and 'is_spoof' in each item.
        n_speakers_per_batch (int): Number of unique speakers per batch.
        n_genuine_per_speaker (int): Number of genuine utterances per speaker.
        n_fake_per_speaker (int): Number of deepfake utterances per speaker.
        drop_incomplete_speakers (bool): If True, drop speakers without both genuine and deepfake samples (default: True).
    """

    def __init__(self, dataset, n_speakers_per_batch, n_genuine_per_speaker, n_fake_per_speaker, drop_incomplete_speakers=True):
        self.dataset = dataset
        self.n_speakers_per_batch = n_speakers_per_batch
        self.n_genuine_per_speaker = n_genuine_per_speaker
        self.n_fake_per_speaker = n_fake_per_speaker
        self.drop_incomplete_speakers = drop_incomplete_speakers

        # Calculate number of batches per epoch
        self.batch_size = n_speakers_per_batch * (n_genuine_per_speaker + n_fake_per_speaker)
        self.num_samples = len(dataset)
        self.num_batches = self.num_samples // self.batch_size

        # Split indices by speaker and label
        self.speaker_to_genuine = defaultdict(list)
        self.speaker_to_fake = defaultdict(list)
        for idx, data_point in tqdm(enumerate(dataset.data_points), desc="Initialising SpeakerGenuineFakeBalancedSampler"):
            spoof = data_point.get('is_spoof', False)
            if spoof:
                self.speaker_to_fake[data_point['speaker_id']].append(idx)
            else:
                self.speaker_to_genuine[data_point['speaker_id']].append(idx)

        # Determine valid speakers based on availability
        if drop_incomplete_speakers:
            self.valid_speakers = list(set(self.speaker_to_genuine.keys()) & set(self.speaker_to_fake.keys()))  # Intersection
        else:
            self.valid_speakers = list(set(self.speaker_to_genuine.keys()) | set(self.speaker_to_fake.keys()))  # Union

        # Error handling
        if not self.speaker_to_genuine:
            raise ValueError("No genuine samples found for any speaker in the dataset. Use SpeakerBalancedBatchSampler instead.")
        if not self.speaker_to_fake:
            raise ValueError("No deepfake samples found for any speaker in the dataset. Use SpeakerBalancedBatchSampler instead.")

        logger.info(f"Initialised SpeakerGenuineFakeBalancedSampler with parameters:\n{self}")


    def __str__(self):
        return (f"Total samples: {self.num_samples}, "
                f"Number of speakers: {len(self.valid_speakers)}, "
                f"Speakers per batch: {self.n_speakers_per_batch}, "
                f"Genuine per speaker: {self.n_genuine_per_speaker}, "
                f"Fake per speaker: {self.n_fake_per_speaker}, "
                f"Batch size: {self.batch_size}, "
                f"Estimated batches per epoch: {self.num_batches}, "
                f"Drop incomplete speakers: {self.drop_incomplete_speakers}")


    def __iter__(self):
        """
        Yields batches of indices, each batch containing:
        - n_speakers_per_batch unique speakers
        - n_genuine_per_speaker genuine utterances per speaker
        - n_fake_per_speaker deepfake utterances per speaker
        """
        for _ in range(self.num_batches):
            # Sample speakers for this batch
            if len(self.valid_speakers) >= self.n_speakers_per_batch:
                selected_speakers = random.sample(self.valid_speakers, self.n_speakers_per_batch)
            else:
                selected_speakers = random.choices(self.valid_speakers, k=self.n_speakers_per_batch)
                logger.warning("Sampling speakers with replacement to complete batch.")
            
            batch_indices = []

            # For each speaker, sample genuine and deepfake utterances
            for spk in selected_speakers:
                genuine_indices = self.speaker_to_genuine.get(spk, [])
                fake_indices = self.speaker_to_fake.get(spk, [])

                # Handle genuine sampling
                if len(genuine_indices) >= self.n_genuine_per_speaker:
                    selected_genuine = random.sample(genuine_indices, self.n_genuine_per_speaker)
                elif len(genuine_indices) > 0:
                    selected_genuine = random.choices(genuine_indices, k=self.n_genuine_per_speaker)
                    logger.warning(f"Speaker {spk} has insufficient genuine samples. Sampling with replacement.")
                else:   # not possible if drop_incomplete_speakers = True
                    selected_genuine = random.choices(fake_indices, k=self.n_genuine_per_speaker)
                    logger.warning(f"Speaker {spk} has no genuine samples. Using fake data instead.")

                # Handle fake sampling
                if len(fake_indices) >= self.n_fake_per_speaker:
                    selected_fake = random.sample(fake_indices, self.n_fake_per_speaker)
                elif len(fake_indices) > 0:
                    selected_fake = random.choices(fake_indices, k=self.n_fake_per_speaker)
                    logger.warning(f"Speaker {spk} has insufficient deepfake samples. Sampling with replacement.")
                else:   # not possible if drop_incomplete_speakers = True
                    selected_fake = random.choices(genuine_indices, k=self.n_fake_per_speaker)
                    logger.warning(f"Speaker {spk} has no deepfake samples. Using genuine data instead.")

                batch_indices.extend(selected_genuine + selected_fake)

            yield batch_indices


    def __len__(self):
        """Returns the number of batches per epoch."""
        return self.num_batches



class SpeakerGenuineFakeVocoderBalancedSampler(Sampler):
    """
    A PyTorch Sampler that yields batches containing a fixed number of speakers,
    with a fixed number of genuine, deepfake, and vocoder utterances per speaker.

    Args:
        dataset (Dataset): Dataset containing 'speaker_id', 'is_spoof', and 'vocoder_name'  in each item.
        n_speakers_per_batch (int): Number of unique speakers per batch.
        n_genuine_per_speaker (int): Number of genuine utterances per speaker.
        n_fake_per_speaker (int): Number of deepfake utterances per speaker.
        n_vocoder_per_speaker (int): Number of vocoder utterances per speaker.
        drop_incomplete_speakers (bool): If True, drop speakers without all three types (default: True).
    """

    def __init__(self, dataset, n_speakers_per_batch, n_genuine_per_speaker, n_fake_per_speaker, n_vocoder_per_speaker, drop_incomplete_speakers=True):
        self.dataset = dataset
        self.n_speakers_per_batch = n_speakers_per_batch
        self.n_genuine_per_speaker = n_genuine_per_speaker
        self.n_fake_per_speaker = n_fake_per_speaker
        self.n_vocoder_per_speaker = n_vocoder_per_speaker
        self.drop_incomplete_speakers = drop_incomplete_speakers

        self.batch_size = n_speakers_per_batch * (n_genuine_per_speaker + n_fake_per_speaker + n_vocoder_per_speaker)
        self.num_samples = len(dataset)
        self.num_batches = self.num_samples // self.batch_size

        self.speaker_to_genuine = defaultdict(list)
        self.speaker_to_fake = defaultdict(list)
        self.speaker_to_vocoder = defaultdict(list)

        for idx, data_point in tqdm(enumerate(dataset.data_points), desc="Initialising SpeakerGenuineFakeVocoderBalancedSampler"):
            speaker_id = data_point['speaker_id']
            is_spoof = data_point.get('is_spoof', False)
            vocoder_name = data_point.get('vocoder_name') or 'bonafide'

            if not is_spoof and vocoder_name.lower() != 'bonafide' and vocoder_name.lower() != 'unknown':
                self.speaker_to_vocoder[speaker_id].append(idx)
            elif not is_spoof:
                self.speaker_to_genuine[speaker_id].append(idx)
            elif is_spoof:
                self.speaker_to_fake[speaker_id].append(idx)

        # Determine valid speakers
        if drop_incomplete_speakers:
            self.valid_speakers = list(
                set(self.speaker_to_genuine.keys()) &
                set(self.speaker_to_fake.keys()) &
                set(self.speaker_to_vocoder.keys())
            )
        else:
            self.valid_speakers = list(
                set(self.speaker_to_genuine.keys()) |
                set(self.speaker_to_fake.keys()) |
                set(self.speaker_to_vocoder.keys())
            )

        # Error handling
        if not self.speaker_to_genuine:
            raise ValueError("No genuine samples found for any speaker in the dataset. Use SpeakerBalancedBatchSampler instead.")
        if not self.speaker_to_fake:
            raise ValueError("No deepfake samples found for any speaker in the dataset. Use SpeakerBalancedBatchSampler instead.")
        if not self.speaker_to_vocoder:
            raise ValueError("No vocoder samples found for any speaker in the dataset. Use SpeakerGenuineFakeBalancedSampler instead.")

        logger.info(f"Initialized Sampler with parameters:\n{self}")

    def __str__(self):
        return (f"Total samples: {self.num_samples}, "
                f"Number of speakers: {len(self.valid_speakers)}, "
                f"Speakers per batch: {self.n_speakers_per_batch}, "
                f"Genuine per speaker: {self.n_genuine_per_speaker}, "
                f"Fake per speaker: {self.n_fake_per_speaker}, "
                f"Vocoder per speaker: {self.n_vocoder_per_speaker}, "
                f"Batch size: {self.batch_size}, "
                f"Estimated batches per epoch: {self.num_batches}, "
                f"Drop incomplete speakers: {self.drop_incomplete_speakers}")

    def __iter__(self):
        """
        Yields batches of indices, each batch containing:
        - n_speakers_per_batch unique speakers
        - n_genuine_per_speaker genuine utterances per speaker
        - n_fake_per_speaker deepfake utterances per speaker
        - n_vocoder_per_speaker vocoder utterances per speaker
        """
        for _ in range(self.num_batches):
            if len(self.valid_speakers) >= self.n_speakers_per_batch:
                selected_speakers = random.sample(self.valid_speakers, self.n_speakers_per_batch)
            else:
                selected_speakers = random.choices(self.valid_speakers, k=self.n_speakers_per_batch)
                logger.warning("Sampling speakers with replacement.")

            batch_indices = []

            for spk in selected_speakers:
                # Get index pools
                genuine = self.speaker_to_genuine.get(spk, [])
                fake = self.speaker_to_fake.get(spk, [])
                vocoder = self.speaker_to_vocoder.get(spk, [])

                # Sample each type
                def safe_sample(pool, n, label):
                    if len(pool) >= n:
                        return random.sample(pool, n)
                    elif len(pool) > 0:
                        logger.warning(f"Speaker {spk} has insufficient {label} samples. Sampling with replacement.")
                        return random.choices(pool, k=n)
                    else:
                        logger.warning(f"Speaker {spk} has no {label} samples. Sampling from available others.")
                        return random.choices(genuine + fake + vocoder, k=n)  # fallback to mixed pool

                batch_indices.extend(safe_sample(genuine, self.n_genuine_per_speaker, 'genuine'))
                batch_indices.extend(safe_sample(fake, self.n_fake_per_speaker, 'fake'))
                batch_indices.extend(safe_sample(vocoder, self.n_vocoder_per_speaker, 'vocoder'))

            yield batch_indices

    def __len__(self):
        """Returns the number of batches per epoch."""
        return self.num_batches



class GenuineFakeBalancedSampler(Sampler):
    """
    A PyTorch Sampler that yields batches containing a fixed number of genuine and deepfake samples,
    regardless of speaker identity.

    Args:
        dataset (Dataset): Dataset with 'is_spoof' in each item.
        n_genuine (int): Number of genuine samples per batch.
        n_fake (int): Number of deepfake samples per batch.
    """

    def __init__(self, dataset, n_genuine, n_fake):
        self.dataset = dataset
        self.n_genuine = n_genuine
        self.n_fake = n_fake

        # Calculate number of batches per epoch
        self.batch_size = n_genuine + n_fake
        self.num_samples = len(dataset)
        self.num_batches = self.num_samples // self.batch_size

        # Initialise indices
        self.genuine_indices = []
        self.fake_indices = []
        for i, d in tqdm(enumerate(dataset.data_points), desc="Initialising GenuineFakeBalancedSampler"):
            if d.get('is_spoof', False):
                self.fake_indices.append(i)
            else:
                self.genuine_indices.append(i)

        # Error handling
        if not self.fake_indices:
            raise ValueError("No deepfake samples found in the dataset. Use SpeakerBalancedBatchSampler instead.")
        if not self.genuine_indices:
            raise ValueError("No genuine samples found in the dataset. Use SpeakerBalancedBatchSampler instead.")
        
        logger.info(f"Initialised GenuineFakeBalancedSampler with parameters:\n{self}")


    def __str__(self):
        return (f"Total samples: {self.num_samples}, "
                f"Genuine samples: {len(self.genuine_indices)}, "
                f"Deepfake samples: {len(self.fake_indices)}, "
                f"Genuine per batch: {self.n_genuine}, "
                f"Fake per batch: {self.n_fake}, "
                f"Batch size: {self.batch_size}, "
                f"Estimated batches per epoch: {self.num_batches}")


    def __iter__(self):
        """
        Yields batches of indices, each batch containing:
        - n_genuine genuine utterances
        - n_fake deepfake utterances 
        """
        for _ in range(self.num_batches):
            batch_indices = []
            
            if len(self.genuine_indices) >= self.n_genuine:
                genuine_batch = random.sample(self.genuine_indices, self.n_genuine)
            else:
                genuine_batch = random.choices(self.genuine_indices, k=self.n_genuine)
                logger.warning("Sampling genuine with replacement to complete batch.")

            if len(self.fake_indices) >= self.n_fake:
                fake_batch = random.sample(self.fake_indices, self.n_fake)
            else:
                fake_batch = random.choices(self.fake_indices, k=self.n_fake)
                logger.warning("Sampling fake with replacement to complete batch.")

            batch_indices.extend(genuine_batch + fake_batch)

            yield batch_indices


    def __len__(self):
        """Returns the number of batches per epoch."""
        return self.num_batches