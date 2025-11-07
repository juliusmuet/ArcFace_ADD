# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class SimpleAdapter(nn.Module):
    """
    A simple adapter module that normalises the input, applies a two-layer feedforward network 
    with ReLU activation in between, and then normalises the output.

    Args:
        input_dim (int): Dimensionality of the input features.
        output_dim (int): Dimensionality of the output features.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim)
        )

        logger.info(f"Initialised SimpleAdapter with input_dim={input_dim} and output_dim={output_dim}.")

    def forward(self, x):
        """
        Forward pass through the adapter.

        Args:
            x (Tensor): Input tensor with feature dimension input_dim.

        Returns:
            Tensor: Output tensor with feature dimension output_dim.
        """
        return self.adapter(x)
    

class ResidualAdapter(nn.Module):
    """
    A residual adapter module that adds a residual (skip) connection to a transformed version 
    of the input. Includes normalisation before and after the transformation.

    Args:
        input_dim (int): Dimensionality of the input features.
        output_dim (int): Dimensionality of the output features.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        self.skip_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.norm = nn.LayerNorm(output_dim)

        logger.info(f"Initialised ResidualAdapter with input_dim={input_dim} and output_dim={output_dim}.")

    def forward(self, x):
        """
        Forward pass through the adapter.

        Args:
            x (Tensor): Input tensor with feature dimension input_dim.

        Returns:
            Tensor: Output tensor with feature dimension output_dim.
        """
        return self.norm(self.proj(x) + self.skip_proj(x))