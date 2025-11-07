# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torch.nn as nn
from s3prl.nn import S3PRLUpstream, Featurizer
import logging
from utils.model_getter import get_adapter

logger = logging.getLogger(__name__)


class WavLMFrontend(nn.Module):
    def __init__(self, model_name='wavlm', model_args={}, device='cpu'):
        """
        A frontend module for extracting learned audio representations using a pretrained WavLM model
        from the S3PRL toolkit.
        Optionally includes a featurizer to combine outputs from multiple transformer layers.
        Includes a projection layer to map features to a desired output dimension.

        Args:
            'model_name' (str): Name of the pretrained WavLM model to load (default: 'wavlm').
            'model_args' (dict): Dictionary of arguments passed to the WavLM model constructor containing the following optional keys:
                - 'finetune' (bool): Whether WavLM parameters should be fine-tuned during training (default: False).
                - 'output_layers' (list[int] or int): Transformer layers to use for feature extraction (default: None / all layers).
                    If `None` or a list with multiple elements, a featurizer is applied.
                - 'layer_norm' (bool): Whether to apply layer normalisation in the featurizer (default: True).
                - 'output_dim' (int): Dimension of the output feature vectors after projection (default: None).
                - 'adapter' (str): Name of the adapter model to use to map WavLM output to output_dim (default: 'SimpleAdapter').
            device (str or torch.device): The device to run the model on (default: 'cpu').
        """
        super().__init__()
        # Load WavLM model
        self.model_name = model_name
        try:
            self.wavlm_model = S3PRLUpstream(self.model_name)
            self.wavlm_model = self.wavlm_model.to(device)
            self.wavlm_model.upstream.model.feature_grad_mult = 1.0
            logger.info(f"Loaded WavLM model: {self.model_name}.")
        except Exception as e:
             raise RuntimeError(f"Failed to load s3prl upstream model '{self.model_name}': {e}.")

        # Freeze WavLM parameters if finetuning is disabled
        self.wavlm_finetune = model_args.get('finetune', False)
        if not self.wavlm_finetune:
            for param in self.wavlm_model.parameters():
                param.requires_grad = False
            logger.info(f"WavLM model frozen.")
        else:   # Freeze unnecessary parameters for fine-tuning
            for name, param in self.wavlm_model.named_parameters():
                if "mask_emb" in name:
                    param.requires_grad = False
            logger.info(f"WavLM model parameters are trainable.")

        # Get Featurizer parameters
        self.selected_layers = model_args.get('output_layers', None)
        if isinstance(self.selected_layers, int):
            self.selected_layers = [self.selected_layers]
        self.layer_norm = model_args.get('layer_norm', True)
        
        # Get output dimension & check selected output layers
        with torch.no_grad():
            dummy_input = torch.randn(1, 16000).to(device)
            dummy_len = torch.tensor([dummy_input.shape[1]], dtype=torch.long, device=device)
            all_hs, _ = self.wavlm_model(dummy_input, dummy_len)
            single_layer_output = all_hs[-1]
            self.input_linear_dim = single_layer_output.shape[-1]
        
        # Setup Featurizer
        self.use_featurizer = self.selected_layers is None or len(self.selected_layers) != 1
        if self.use_featurizer:
            self.featurizer = Featurizer(self.wavlm_model, layer_selections=self.selected_layers, normalize=self.layer_norm)
            self.featurizer = self.featurizer.to(device)
            self.input_linear_dim = self.featurizer.output_size
            logger.info(f"Using Featurizer with layers {self.selected_layers} (None means all) of WavLM.")
        else:
            logger.info(f"Using only layer {self.selected_layers[0]} of WavLM without Featurizer.")

        # Setup normalisation
        self.layer_norm_layer = nn.LayerNorm(self.input_linear_dim, device=device)

        # Setup Projection Layer 
        self.projection_layer = None
        self.output_dim = model_args.get('output_dim', None)
        if self.output_dim is None:
            self.output_dim = self.input_linear_dim
        else:
            self.projection_layer = get_adapter(model_args.get('adapter', "SimpleAdapter") )(self.input_linear_dim, self.output_dim)
            self.projection_layer = self.projection_layer.to(device)

        logger.info(f"Initialised WavLM Frontend with parameters:\n{self}")


    def __str__(self):
        base_str = (
        f"WavLMFrontend(model_name='{self.model_name}', "
        f"device='{self._get_module_device()}', "
        f"finetune={self.wavlm_finetune}, "
        f"selected_layers={self.selected_layers}, "
        f"layer_norm={self.layer_norm}, "
        f"use_featurizer={self.use_featurizer}"
        )

        if self.projection_layer is not None:
            base_str += (
                f", adapter={self.projection_layer.__class__.__name__}, "
                f"input_linear_dim={self.input_linear_dim}"
            )

        base_str += f", output_dim={self.output_dim})"
        return base_str


    def forward(self, wavs):
        """
        Forward pass of the WavLMFrontend.

        Args:
            wavs (torch.Tensor): A tensor of shape (batch_size, seq_len) or (seq_len,) containing raw waveform samples.
                                 If 1D, it is reshaped to (1, seq_len).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, num_frames, output_dim) containing the projected WavLM features.
        """
        # Reshape (seq_len,) to (1, seq_len)
        if wavs.dim() == 1:
            wavs = wavs.unsqueeze(0)
        
        if wavs.dim() != 2:
            raise ValueError(f"Expected shape of wavs to be (seq_len,) or (batch_size, seq_len), but got shape {wavs.shape}")
        
        # Move input to device
        device = self._get_module_device()
        wavs = wavs.to(device)
    
        wavs_len = torch.tensor([wav.shape[0] for wav in wavs], dtype=torch.long, device=device)    # Create input length tensor
        
        if not self.wavlm_finetune:
            with torch.no_grad():   # Disable gradient computation if no finetuning
                all_hs, all_hs_len = self.wavlm_model(wavs, wavs_len)
        else:
            all_hs, all_hs_len = self.wavlm_model(wavs, wavs_len)

        if self.use_featurizer:
            hs, _ = self.featurizer(all_hs, all_hs_len)
        else:
            hs = all_hs[self.selected_layers[0]]

        hs = self.layer_norm_layer(hs)

        if self.projection_layer is not None:
            return self.projection_layer(hs)    # Shape (batch_size, num_frames, output_dim)
        else:
            return hs    # Shape (batch_size, num_frames, output_dim)
    

    def _get_module_device(self, module=None):
        """
        Returns the device of the first parameter of the given module.
        This is useful for determining the current device a module is on,
        especially when modules are moved between devices using `.to(device)`.

        Args:
            module (torch.nn.Module, optional): The module whose device should be determined.
                If None, `self` is used.

        Returns:
            torch.device: The device on which the module's parameters are currently located.
        """
        if module is None:
            module = self
        
        return next(module.parameters()).device
