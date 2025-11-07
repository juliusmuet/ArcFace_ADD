# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from pathlib import Path
import torch
import torch.nn as nn
import logging
from frontends.wavlm import WavLMFrontend
from utils.model_getter import get_backend_model, get_classifier_model

logger = logging.getLogger(__name__)


class SpeakerEmbedderModel(nn.Module):
    def __init__(self, backend_config, frontend_config=None, classifier_config=None, checkpoint_path=None, device='cpu'):
        """
        A modular speaker embedding model that can include a WavLM-based frontend,
        an embedding backend, and an optional classifier head.

        Args:
            backend_config (dict): Configuration for the embedding backend containing:
                - 'model_name' (str): Identifier for the backend model (used by get_backend_model()).
                - 'model_args' (dict): Keyword arguments to instantiate the backend model.
                - 'checkpoint_path' (str, optional): Path to a pretrained backend weights file.

            frontend_config (dict, optional): Configuration for the WavLM frontend containing:
                - 'model_name' (str): Identifier for the WavLM frontend model (used by WavLMFrontend).
                - 'model_args' (dict): Keyword arguments to instantiate the WavLM frontend model.

            classifier_config (dict, optional): Configuration for the classifier head containing:
                - 'model_name' (str): Identifier for the classifier model (used by get_classifier_model()).
                - 'model_args' (dict): Keyword arguments to instantiate the classifier model.
                - 'loss' (dict): Not needed in this class but in factory.py.

            checkpoint_path (str, optional): Path to a full model checkpoint to load. If provided,
                the full model (frontend, backend, classifier) will be loaded from this file.

            device (str or torch.device): The device to run the model on (default: 'cpu').

        Notes:
            - If `frontend_config` is None, the model expects pre-computed features instead of raw audio.
            - If `classifier_config` is None, only embeddings will be returned during forward passes.
        """
        super().__init__()
        # Setup WavLM frontend
        self.frontend = None
        if frontend_config is not None:
            self.frontend = WavLMFrontend(**frontend_config)
        else:
            logger.info("WavLM Frontend disabled. Model expects pre-computed features as input.")

        # Setup backend
        if self.frontend is not None:
            backend_config.get('model_args', {})['feat_dim'] = self.frontend.output_dim
        self.backend_input_dim = backend_config.get('model_args', {}).get('feat_dim', 80)
        self.backend = get_backend_model(backend_config.get('model_name'))(**backend_config.get('model_args', {}))
        if backend_config.get('checkpoint_path', None) is not None:
            state_dict = torch.load(Path(backend_config['checkpoint_path']), map_location=self._get_module_device(self.backend))
            model_dict = self.backend.state_dict()
            new_state_dict = {}
            for k, v in state_dict.items():
                if k in model_dict:
                    if model_dict[k].shape == v.shape:
                        new_state_dict[k] = v
                    else:
                        logger.info(f"Shape mismatch for {k}: checkpoint shape = {v.shape}, model shape = {model_dict[k].shape}")
            self.backend.load_state_dict(new_state_dict, strict=False)
            logger.info(f"Initialised pretrained {backend_config.get('model_name')} Backend from {Path(backend_config['checkpoint_path'])} with parameters {backend_config}.")
        else:
            logger.info(f"Initialised {backend_config.get('model_name')} Backend with parameters {backend_config}.")

        # Setup classifier
        self.classifier = None
        if classifier_config is not None:
            self.classifier = get_classifier_model(classifier_config.get('model_name'))(**classifier_config.get('model_args', {}))
        else:
            logger.info("No classifier initialised.")

        # Move whole model (with submodels) to device
        self.to(device)

        # Optional: load parameters from checkpoint
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        logger.info(f"Initialised Speaker Embedding Model on device {device}.") 


    def forward(self, inputs):
        """
        Forward pass of the Speaker Verification Model.

        Args:
            inputs (torch.Tensor):
                - If frontend is used: Raw audio waveforms of shape (batch_size, seq_len).
                - If frontend is NOT used: Pre-computed features of shape
                  (batch_size, num_frames, feature_dim) expected by the backend.

        Returns:
            torch.Tensor: Speaker embeddings of shape (batch_size, embedding_dim).
            torch.Tensor or None: Logits of shape (batch_size, num_classes) if the
                                  classifier is used, otherwise None.
        """
        device = self._get_module_device()
        inputs = inputs.to(device)

        if self.frontend:       
            features = self.frontend(inputs)    # Shape (batch_size, num_frames, frontend_output_dim=backend_input_dim)
        else:
            features = inputs

        outputs = self.backend(features) # Shape (batch_size, embedding_dim)
        embeddings = outputs[-1] if isinstance(outputs, tuple) else outputs # Handle models that have two outputs (e.g. WeSpeaker ECAPA-TDNN)

        logits = None
        if self.classifier:
            logits = self.classifier(embeddings)
            return embeddings, logits
        else:
            return embeddings, None


    def get_trainable_parameters_backend(self):
        """
        Returns all trainable parameters except the core WavLM SSL model parameters.
        """
        all_params = list(filter(lambda p: p.requires_grad, self.parameters()))

        # Remove SSL params from the full parameter set
        ssl_params = set(self.get_trainable_parameters_ssl())
        backend_params = [p for p in all_params if p not in ssl_params]

        return backend_params


    def get_trainable_parameters_ssl(self):
        """
        Returns only the trainable parameters of the core WavLM SSL model.
        """
        if self.frontend is None or not hasattr(self.frontend, "wavlm_model"):
            return []

        ssl_params = list(filter(lambda p: p.requires_grad, self.frontend.wavlm_model.parameters()))
        return ssl_params
    

    def save_checkpoint(self, path):
        """
        Saves the model state dict to the specified path.

        Args:
            path (str): Path to save the checkpoint.
        """
        torch.save(self.state_dict(), path)
        logger.info(f"Model checkpoint saved to {path}")


    def load_checkpoint(self, path):
        """
        Loads model weights from the specified checkpoint file.

        Args:
            path (str): Path to the checkpoint file.
        """
        device = self._get_module_device()
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'], strict=True)
        logger.info(f"Model checkpoint loaded from {path}")

    
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
    

    def get_backend_embedding_dim(self):
        """
        Returns:
            int: Dimension of embedding output by the backend model.
        """
        with torch.no_grad():
            dummy_input = torch.randn(2, 10, self.backend_input_dim).to(self._get_module_device())
            emb = self.backend(dummy_input)
            emb = emb[-1] if isinstance(emb, tuple) else emb # Handle models that have two outputs (e.g. WeSpeaker ECAPA-TDNN)
            return emb.shape[-1]
