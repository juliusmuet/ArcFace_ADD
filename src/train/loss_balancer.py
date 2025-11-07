# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class WeightedSum(nn.Module):
    def __init__(self, use_deepfake, use_classifier, sigma_genuine=1.0, sigma_deepfake=1.0, sigma_classifier=1.0):
        """
        Implements multi-task loss using weighted sum with fixed parameters.

        Args:
            use_deepfake (bool): Whether to include the deepfake loss in the total loss computation.
            use_classifier (bool): Whether to include the classifier loss in the total loss computation.
            sigma_genuine (float): Multiplicative parameter for the genuine embedding loss (default: 1.0).
            sigma_deepfake (float): Multiplicative parameter for the deepfake embedding loss (default: 1.0).
            sigma_classifier (float): Multiplicative parameter for the classifier loss (default: 1.0).
        """
        super().__init__()
        self.use_deepfake = use_deepfake
        self.use_classifier = use_classifier
        self.sigma_genuine = sigma_genuine
        self.sigma_deepfake = sigma_deepfake
        self.sigma_classifier = sigma_classifier

        logger.info(f"Initialised WeightedSum Loss Balancer with parameters: \n{self}")

    def __str__(self):
        return (
            f"WeightedSum(use_deepfake: {self.use_deepfake}, "
            f"use_classifier: {self.use_classifier}, "
            f"sigma_genuine: {self.sigma_genuine}, "
            f"sigma_deepfake: {self.sigma_deepfake}, "
            f"sigma_classifier: {self.sigma_classifier})"
        )

    def get_genuine_parameter(self):
        return self.sigma_genuine

    def get_deepfake_parameter(self):
        return self.sigma_deepfake
    
    def get_classifier_parameter(self):
        return self.sigma_classifier
    
    def forward(self, loss_genuine, loss_deepfake=None, loss_classifier=None):
        """
        Computes the total loss using weighted sum with fixed parameters.

        Args:
            loss_genuine (Tensor): The loss associated with the genuine embedding loss.
            loss_deepfake (Tensor, optional): The loss associated with the deepfake embedding loss (default: None).
            loss_classifier (Tensor, optional): The loss associated with the classifier task (default: None).

        Returns:
            Tensor: The total weighted loss.
        """
        total = self.sigma_genuine * loss_genuine

        if self.use_deepfake and loss_deepfake is not None:
            total += self.sigma_deepfake * loss_deepfake

        if self.use_classifier and loss_classifier is not None:
            total += self.sigma_classifier * loss_classifier

        return total        


class UncertaintyWeighting(nn.Module):
    def __init__(self, use_deepfake, use_classifier):
        """
        Implements multi-task loss weighting using learned uncertainty, as introduced in the paper:
        "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics" 
        by Kendall et al.

        This module learns individual weights for multiple loss components (genuine, deepfake, and classifier losses)
        based on their task uncertainty, enabling the model to adaptively balance the influence of each loss term.

        Args:
            use_deepfake (bool): Whether to include the deepfake loss in the total loss computation.
            use_classifier (bool): Whether to include the classifier loss in the total loss computation.
        """
        super().__init__()
        self.use_deepfake = use_deepfake
        self.use_classifier = use_classifier

        self.log_sigma_genuine = nn.Parameter(torch.tensor(0.0))

        if use_deepfake:
            self.log_sigma_deepfake = nn.Parameter(torch.tensor(0.0))
        else:
            self.log_sigma_deepfake = None

        if use_classifier:
            self.log_sigma_classifier = nn.Parameter(torch.tensor(0.0))
        else:
            self.log_sigma_classifier = None

        logger.info(f"Initialised UncertaintyWeighting Loss Balancer with parameters: \n{self}")

    def __str__(self):
        return (
            f"UncertaintyWeighting(use_deepfake: {self.use_deepfake}, "
            f"use_classifier: {self.use_classifier}, "
            f"sigma_genuine: {self.get_genuine_parameter()}, "
            f"sigma_deepfake: {self.get_deepfake_parameter()}, "
            f"sigma_classifier: {self.get_classifier_parameter()})"
        )

    def get_genuine_parameter(self):
        if self.log_sigma_genuine is None:
            return 0.0
        return self.log_sigma_genuine.item()

    def get_deepfake_parameter(self):
        if self.log_sigma_deepfake is None:
            return 0.0
        return self.log_sigma_deepfake.item()
    
    def get_classifier_parameter(self):
        if self.log_sigma_classifier is None:
            return 0.0
        return self.log_sigma_classifier.item()
    
    def forward(self, loss_genuine, loss_deepfake=None, loss_classifier=None):
        """
        Computes the total loss using learned uncertainty-based weighting.

        Each individual loss is weighted by the inverse of its uncertainty, allowing the model to
        down-weight noisy or difficult tasks during training.

        Args:
            loss_genuine (Tensor): The loss associated with the genuine embedding loss.
            loss_deepfake (Tensor, optional): The loss associated with the deepfake embedding loss (default: None).
            loss_classifier (Tensor, optional): The loss associated with the classifier task (default: None).

        Returns:
            Tensor: The total weighted loss.
        """
        if not self.use_deepfake and not self.use_classifier:
            return loss_genuine

        total = torch.exp(-self.log_sigma_genuine) * loss_genuine + self.log_sigma_genuine

        if self.use_deepfake and loss_deepfake is not None:
            total += torch.exp(-self.log_sigma_deepfake) * loss_deepfake + self.log_sigma_deepfake

        if self.use_classifier and loss_classifier is not None:
            total += torch.exp(-self.log_sigma_classifier) * loss_classifier + self.log_sigma_classifier

        return total