# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import pickle
from tqdm import tqdm


class BaseEncoder:
    """
    Base class providing shared functionality for label encoders.

    Args:
        lab2ind (dict): Mapping from string labels to integer indices.
        ind2lab (dict): Mapping from integer indices to string labels.
        starting_index (int): Counter to assign the next available index.
    """

    def __init__(self):
        self.lab2ind = {}
        self.ind2lab = {}
        self.starting_index = -1

    def __len__(self):
        """
        Returns:
            int: Number of unique labels.
        """
        return len(self.lab2ind)

    def _add(self, label):
        """
        Adds a new string label to the encoder if not already present.

        Args:
            label (str): Label to be added.
        """
        if label in self.lab2ind:
            return
        index = self._next_index()
        self.lab2ind[label] = index
        self.ind2lab[index] = label

    def _next_index(self):
        """
        Returns:
            int: Next available index for a new label.
        """
        self.starting_index += 1
        return self.starting_index

    def save(self, path):
        """
        Saves the label-to-index mapping to a file.

        Args:
            path (str): Path to the output file.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.lab2ind, f)

    def load(self, path):
        """
        Loads the label-to-index mapping from a file.

        Args:
            path (str): Path to the file containing the mapping.
        """
        self.lab2ind = {}
        self.ind2lab = {}
        with open(path, 'rb') as f:
            self.lab2ind = pickle.load(f)
        for label in self.lab2ind:
            self.ind2lab[self.lab2ind[label]] = label


class SpkLabelEncoder(BaseEncoder):
    """
    Encoder for mapping string speaker labels to unique integer indices.

    Args:
        speaker_names (list[str]): List of speaker IDs.
        speakers_with_deepfakes (list[str]): Unused, included for interface consistency (default: None).
    """

    def __init__(self, speaker_names, speakers_with_deepfakes=None):
        super().__init__()
        self.speaker_names = sorted(speaker_names)
        self._load_from_speaker_names()

    def __call__(self, spk, genuine = 1):
        """
        Returns the index for a given speaker.

        Args:
            spk (str): Speaker ID.
            genuine (int): Unused, included for interface consistency (default: 1).

        Returns:
            int: Encoded speaker index.
        """
        return self.lab2ind[spk]

    def _load_from_speaker_names(self):
        """Adds all speaker names to the encoder."""
        for spk in tqdm(self.speaker_names, desc="Initialising SpkLabelEncoder"):
            self._add(spk)


class GroupedGenuineDeepfakePairEncoder(BaseEncoder):
    """
    Encoder that assigns unique indices to each speaker's genuine and deepfake versions.
    All genuine speakers are indexed first, followed by all their deepfake counterparts.

    Args:
        genuine_speakers (list[str]): List of genuine speaker IDs.
        speakers_with_deepfakes (list[str]): List of speaker IDs with deepfakes.
    """

    def __init__(self, genuine_speakers, speakers_with_deepfakes):
        super().__init__()
        self.genuine_speakers = sorted(genuine_speakers)
        self.speakers_with_deepfakes = sorted(speakers_with_deepfakes)
        self._load_from_speaker_names()

    def __call__(self, spk, genuine):
        """
        Returns the index for a speaker depending on whether it's genuine or deepfake.

        Args:
            spk (str): Base speaker ID.
            genuine (int): 1 if genuine, 0 if deepfake.

        Returns:
            int: Encoded speaker index.
        """
        if genuine == 1:
            return self.lab2ind[spk]
        else:
            return self.lab2ind[f"{spk}_df"]

    def _load_from_speaker_names(self):
        """Adds all genuine and deepfake speaker labels to the encoder."""
        for spk in tqdm(self.genuine_speakers, desc="Initialising GroupedGenuineDeepfakePairEncoder Genuine"):
            self._add(spk)
        for spk in tqdm(self.speakers_with_deepfakes, desc="Initialising GroupedGenuineDeepfakePairEncoder Deepfake"):
            self._add(f"{spk}_df")
        

class AlternatingGenuineDeepfakePairEncoder(BaseEncoder):
    """
    Encoder that assigns indices to each speaker's genuine and deepfake versions,
    alternating between genuine and deepfake. If there are deepfake only speakers,
    these will be added at the end.

    Args:
        genuine_speakers (list[str]): List of genuine speaker IDs.
        speakers_with_deepfakes (list[str]): List of speaker IDs with deepfakes.
    """

    def __init__(self, genuine_speakers, speakers_with_deepfakes):
        super().__init__()
        self.genuine_speakers = sorted(genuine_speakers)
        self.speakers_with_deepfakes = sorted(speakers_with_deepfakes)
        self._load_from_speaker_names()

    def __call__(self, spk, genuine):
        """
        Returns the index for a speaker depending on whether it's genuine or deepfake.

        Args:
            spk (str): Base speaker ID.
            genuine (int): 1 if genuine, 0 if deepfake.

        Returns:
            int: Encoded speaker index.
        """
        if genuine == 1:
            return self.lab2ind[spk]
        else:
            return self.lab2ind[f"{spk}_df"]

    def _load_from_speaker_names(self):
        """Adds speaker labels in alternating genuine-deepfake order."""
        seen = set()

        for spk in tqdm(self.genuine_speakers, desc="Initialising AlternatingGenuineDeepfakePairEncoder"):
            self._add(spk)
            seen.add(spk)
            if spk in self.speakers_with_deepfakes:
                self._add(f"{spk}_df")
                seen.add(f"{spk}_df")

        for spk in tqdm(self.speakers_with_deepfakes, desc="Initialising AlternatingGenuineDeepfakePairEncoder Deepfake-Only Speakers"):
            if spk not in seen:
                self._add(f"{spk}_df")


class DeepfakeUnifiedEncoder(BaseEncoder):
    """
    Encoder that assigns a unique label to each genuine speaker,
    and a shared label for all deepfake speakers.

    Args:
        genuine_speakers (list[str]): List of genuine speaker IDs.
        speakers_with_deepfakes (list[str]): Unused, included for interface consistency (default: None).
    """

    def __init__(self, genuine_speakers, speakers_with_deepfakes=None):
        super().__init__()
        self.genuine_speakers = sorted(genuine_speakers)
        self.deepfake_label: str = "deepfake"
        self._load_from_speaker_names()

    def __call__(self, spk, genuine):
        """
        Returns the index for a speaker depending on whether it's genuine or deepfake.

        Args:
            spk (str): Base speaker ID.
            genuine (int): 1 if genuine, 0 if deepfake.

        Returns:
            int: Encoded speaker index.
        """
        if genuine == 1:
            return self.lab2ind[spk]
        else:
            return self.lab2ind[self.deepfake_label]

    def _load_from_speaker_names(self):
        """Adds speaker labels and the shared deepfake label."""
        for spk in tqdm(self.genuine_speakers, desc="Initialising DeepfakeUnifiedEncoder"):
            self._add(spk)
        self._add(self.deepfake_label)
