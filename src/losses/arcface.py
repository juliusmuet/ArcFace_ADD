# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class ArcFace(nn.Module):  
    def __init__(self, emb_dim, num_classes, scale=32.0, margin=0.2, use_genuine_labels=False, easy_margin=False, criterion=None, device='cpu'):
        """
        Implements the ArcFace loss function as a PyTorch module.
        
        ArcFace introduces an additive angular margin penalty between embeddings 
        and class centers to enhance discriminative power of features.

        Args:
            emb_dim (int): Size of each input sample (embedding dimension).
            num_classes (int): Number of output classes.
            scale (float): Scaling factor applied to the logits (default: 32.0).
            margin (float): Angular margin added to the target class (default: 0.2).
            use_genuine_labels (bool): Whether to use the custom deepfake variant of ArcFace (default: False).
            easy_margin (bool): Whether to use the 'easy margin' strategy (default: False).
            criterion (nn.Module): Loss function for the ArcFace logits (default: None).
                                   If None, torch.nn.CrossEntropyLoss will be used.
            device (str): Device identifier (e.g., 'cpu' or 'cuda') specifying where the model should run.

        Notes:
            - use_genuine_labels must only be used with SpkLabelEncoder !!!
            - If use_genuine_labels then
                - for genuine inputs ArcFace is computed as usual
                - for deepfake inputs the probability for the corresponding speaker class is decreased 
                  by forcing the deepfake embeddings to move (not the class centres)
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.use_genuine_labels = use_genuine_labels
        self.easy_margin = easy_margin
        self.criterion = criterion or nn.CrossEntropyLoss(reduction='none')

        # Weight parameter representing class centers
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, emb_dim))
        nn.init.xavier_uniform_(self.weight)

        # Precompute constants for margin adjustment
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.mmm = 1.0 + math.cos(math.pi - margin) # Used in WeSpeaker & 3D-Speaker

        self.to(device)

        logger.info(f"Initialised ArcFace Projection with parameters:\n{self}")


    def __str__(self):
        return (
            f"ArcFace(emb_dim={self.emb_dim}, "
            f"num_classes={self.num_classes}, "
            f"scale={self.scale}, "
            f"margin={self.margin}, "
            f"use_genuine_labels={self.use_genuine_labels}, "
            f"easy_margin={self.easy_margin}, "
            f"criterion={self.criterion.__class__.__name__}, "
            f"criterion.reduction={self.criterion.reduction})"
        )
    

    def update(self, margin=0.2):
        """
        Updates the angular margin value and its derived constants.

        Args:
            margin (float): New margin to be applied.
        """      
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.mmm = 1.0 + math.cos(math.pi - margin)

    
    def get_margin(self):
        """
        Returns the current angular margin value.
        """
        return self.margin


    def forward(self, embeddings, labels, genuine_labels=None):
        """
        Computes the modified logits with the ArcFace angular margin.

        Args:
            embeddings (torch.Tensor): A tensor of shape (batch_size, seq_len) or (seq_len,) containing embedding vectors.
                                       If 1D, it is reshaped to (1, seq_len).
            labels (torch.Tensor): Ground truth class labels of shape (batch_size,).
            genuine_labels (torch.Tensor): The genuine flags for the corresponding inputs (1 for genuine, 0 for spoof) of shape (batch_size,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - A scalar tensor representing the genuine loss value.
                - A scalar tensor representing the deepfake loss value or None if use_genuine_labels is False.
        """ 
        # Reshape (seq_len,) to (1, seq_len)
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        if embeddings.dim() != 2:
            raise ValueError(f"Expected shape of embeddings to be (seq_len,) or (batch_size, seq_len), but got shape {embeddings.shape}")
        if labels.dim() != 1:
            raise ValueError(f"Expected shape of labels to be (batch_size,), but got shape {labels.shape}")
               
        # Cosine similarity cos(θ) between embeddings and class weight vectors
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))    # Shape (batch_size, num_classes)

        # Compute sin(θ) from cos(θ) using the identity: sin(θ) = sqrt(1 - cos²(θ))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # Angular margin for adjusted logits cos(θ + m) = cos(θ) ⋅ cos(m) − sin(θ) ⋅ sin(m)
        # Calculating θ = arccos(θ) and then cos(θ + m) can be numerically unstable
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            # Apply margin only if θ < pi/2 / angle is less than 90° to class centre
            # Else do not apply margin as embedding is already far away from class centre (angle greater than 90°) for better training stability
            phi = torch.where(cosine > 0, phi, cosine)  
        else:
            # Apply margin only if θ < pi-m / θ+m < pi
            # Because else cos(θ + m) increases and larger angle (worse fit) would mean higher score / cos-similarity due to non-monotonicity of cos
            # Else decrease score / cos-similarity for punishment
            phi = torch.where(cosine > self.th, phi, cosine - self.mmm) # or self.mm instead of self.mmm

        # Generate one-hot labels to identify the ground-truth class per sample
        one_hot = embeddings.new_zeros(cosine.size())
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)    # For each row/sample i, put 1 at position label[i]

        # Use phi (cos(θ + m)) for the true class.
        # Use unmodified cos(θ) for other (non-target) classes.
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # Multiply by scaling factor
        output *= self.scale    # Shape (batch_size, num_classes)

        # Standard ArcFace 
        if not self.use_genuine_labels:
            # Calculate loss
            loss = self.criterion(output, labels).mean()
            return loss, None
        
        # ----- Custom ArcFace for deepfakes ----- 
        # Create boolean masks for correct averaging
        genuine_mask = genuine_labels.bool()
        deepfake_mask = ~genuine_mask

        # Genuine loss
        loss_genuine = torch.tensor(0.0, device=embeddings.device)
        num_genuine = genuine_mask.sum()
        if num_genuine > 0:
            loss_genuine = self.criterion(output[genuine_mask], labels[genuine_mask]).mean()

        # Deepfake loss
        loss_deepfake = torch.tensor(0.0, device=embeddings.device)
        num_deepfake = deepfake_mask.sum()
        if num_deepfake > 0:
            # Detach the weights / class centres as these should not change due to deepfake embeddings but only the deepfake embeddings should change
            cosine_detach = F.linear(F.normalize(embeddings[deepfake_mask]), F.normalize(self.weight.detach()))
            
            # Extract deepfaked speaker logits / cosine similarities
            deepfake_labels = labels[deepfake_mask]
            target_cosine = cosine_detach.gather(1, deepfake_labels.view(-1, 1)).squeeze(1)

            # Map [-1, 1] range to [0, 2], ensuring a non-negative loss
            loss_deepfake = (1.0 + target_cosine).mean()
        
        return loss_genuine, loss_deepfake
