# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class LengthBasedClassifier(nn.Module):
    def __init__(self):
        """
        A PyTorch classifier that outputs a logit based solely on the
        L2 norm (length) of an input embedding vector.

        The model consists of a single linear layer that takes the L2 norm of 
        the embedding (a scalar) as input and outputs a logit.
        """
        super().__init__()
        self.fc = nn.Linear(1, 1)   # Maps the length to a logit

        logger.info("Initialised Length Based Classifier.")
        

    def forward(self, embeddings):
        """
        Forward pass of the model.

        Args:
            embeddings (torch.Tensor): A tensor of shape (batch_size, seq_len) or (seq_len,) containing embedding vectors.
                                       If 1D, it is reshaped to (1, seq_len).

        Returns:
            torch.Tensor: A tensor of shape (batch_size,) containing logits computed based on the input vector lengths.
        """
        # Reshape (seq_len,) to (1, seq_len)
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        if embeddings.dim() != 2:
            raise ValueError(f"Expected shape of embs to be (seq_len,) or (batch_size, seq_len), but got shape {embeddings.shape}")
        
        # Compute the length (L2 norm) of each embedding vector in the batch
        lengths = torch.linalg.vector_norm(embeddings, dim=1, keepdim=True)  # Shape: (batch_size, 1)
        
        # Pass length through linear layer
        return self.fc(lengths).squeeze(-1)  # Shape: (batch_size,)
