# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import copy
from pathlib import Path
from itertools import chain
import logging
import torch
import utils.config_loader as config_loader
import utils.model_getter as model_getter
from preprocessing.wav_loader import WavLoader
from speaker_embedder.speaker_embedder import SpeakerEmbedderModel
from datasets.dataset_df import ASVDataset_DF
from datasets.dataset_evaluation import TrialDataset
from utils.utils import set_seed, get_num_accumulation_steps_per_epoch
from train.lr_scheduler import SchedulerWithFixedLR

logger = logging.getLogger(__name__)


class Factory:
    """
    Factory class responsible for creating and initialising all components of a speaker embedding pipeline.
    It loads configuration from a YAML file and uses it to instantiate preprocessors, models, loss functions,
    optimisers and schedulers.

    Args:
        config_path (str): Path to the configuration YAML file.
        checkpoint_path (str): Path to a model checkpoint file (default: None). Used for testing and inference.
    """

    def __init__(self, config_path, checkpoint_path=None):
        self.base_config = config_loader.load_config_file(Path("src/configs/config.yml")) # Used for file paths to datasets
        self.config_directory = Path(config_path).parent
        self.config = config_loader.load_config_file(Path(config_path))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.preprocessor = None
        self.preprocessor_evaluation = None
        self.train_dataset = None
        self.validation_datasets = None
        self.validation_once_datasets = None
        self.test_datasets = None
        self.train_dataloader = None
        self.speaker_embedder = None
        self.embedding_projection = None
        self.margin_scheduler = None
        self.embedding_loss = None
        self.classifier_loss = None
        self.optimiser = None
        self.lr_scheduler = None
        self.loss_balancer = None
        self.steps_per_epoch = None
        self._load_all_configs()


    def _load_all_configs(self):
        """
        Loads all relevant configuration sections from the config file.
        Adjusts the preprocessing and frontend configuration if the frontend is FBank.
        """
        self.seed = self.base_config.get('seed', 42)
        self.config_train = config_loader.get_train_config(self.config)
        self.config_loss_balancer = config_loader.get_loss_balancer_config(self.config)
        self.config_train_dataset = config_loader.get_train_dataset_config(self.config)
        self.config_train_dataloader = config_loader.get_train_dataloader_config(self.config)
        self.config_validation = config_loader.get_validation_config(self.config)
        self.config_validation_dataset = config_loader.get_validation_dataset_config(self.config)
        self.config_test = config_loader.get_test_config(self.config)
        self.config_test_dataset = config_loader.get_test_dataset_config(self.config)
        self.config_preprocessing = config_loader.get_preprocessing_config(self.config)
        self.config_frontend = config_loader.get_frontend_config(self.config)
        self.config_backend = config_loader.get_backend_config(self.config)
        self.config_embedding_projection = config_loader.get_embedding_projection_config(self.config)
        self.config_margin_scheduler = config_loader.get_margin_scheduler_configs(self.config)
        self.config_classifier = config_loader.get_classifier_config(self.config)
        self.config_optimiser = config_loader.get_optimiser_config(self.config)
        self.config_lr_scheduler = config_loader.get_lr_scheduler_config(self.config)

        # if FBank frontend, pass FBank parameters to preprocessing config
        if self.config_frontend.get('model_name').lower() == "fbank":
            self.config_backend.get('model_args', {})['feat_dim'] = self.config_frontend.get('model_args', {}).get('n_mels', 80)
            self.config_preprocessing['fbank_config'] = self.config_frontend.get('model_args', {})
            self.config_frontend = None

        # set seed
        set_seed(self.seed)


    def create_preprocessor(self):
        """
        Creates and returns the audio preprocessing module.
        
        Returns:
            WavLoader: The initialised wavloader.
        """
        if self.preprocessor is not None:
            return self.preprocessor

        self.preprocessor = WavLoader(self.config_preprocessing)
        return self.preprocessor
    

    def create_preprocessor_evaluation(self):
        """
        Creates and returns the audio preprocessing module for evaluation.
        
        Returns:
            WavLoader: The initialised wavloader.
        """
        if self.preprocessor_evaluation is not None:
            return self.preprocessor_evaluation
        
        wav_loader_config = copy.deepcopy(self.config_preprocessing)
        wav_loader_config['duration'] = -1.0
        
        self.preprocessor_evaluation = WavLoader(wav_loader_config)
        return self.preprocessor_evaluation
    

    def create_train_dataloader(self, wav_loader=None):
        """
        Creates and returns the audio dataloader for the train dataset.
        
        Returns:
            WavLoader: The initialised wavloader.
        """
        if self.train_dataloader is not None:
            return self.train_dataloader
        
        if wav_loader is None:
            if self.preprocessor is None:
                self.preprocessor = self.create_preprocessor()
            wav_loader = self.preprocessor
        
        self.config_train_dataset['wav_loader'] = wav_loader
        self.train_dataset = ASVDataset_DF(**self.config_train_dataset)
        self.train_dataloader = self.train_dataset.get_dataloader(self.config_train_dataloader.get('dataloader_args', {}), sampler_config=self.config_train_dataloader.get('sampler_args'))
        return self.train_dataloader
    

    def _create_evaluation_datasets(self, dataset_config):
        """
        Creates and returns a list of evaluation datasets from the given configuration.
        If multiple trial dataset names are provided (space-separated), a TrialDataset
        will be created for each.

        Args:
            dataset_config (dict): Configuration for the dataset, including the key
                'trial_dataset' which can contain one or more dataset names.

        Returns:
            List[TrialDataset]: A list of TrialDataset instances created from the specified configurations.
        """
        trial_datasets = []

        for name in dataset_config.get('trial_dataset').split():
            trial_config = self.base_config.get('deepfake_speaker_verification_evaluation').get(name)
            
            updated_config = dataset_config.copy()
            for k, v in trial_config.items():
                updated_config[k] = Path(v) if isinstance(v, str) else v

            config = {**trial_config, **{k: v for k, v in updated_config.items() if k != 'trial_dataset'}}
            config['wav_loader'] = self.create_preprocessor_evaluation()

            trial_datasets.append(TrialDataset(**config))

        return trial_datasets
    

    def create_validation_datasets(self):
        """
        Creates and returns a list of validation datasets from the given configuration.

        Returns:
            List[TrialDataset] or None: A list of TrialDataset instances, or None if not configured.
        """
        if self.validation_datasets is not None:
            return self.validation_datasets
        
        if self.config_validation_dataset is None:
            logger.info("No validation dataset configured.")
            return None
        
        # Remove once_dataset key temporarily as it is not used here
        config_copy = copy.deepcopy(self.config_validation_dataset)
        config_copy.pop('once_dataset', None)

        self.validation_datasets = self._create_evaluation_datasets(config_copy)
        return self.validation_datasets
    

    def create_validation_once_datasets(self):
        """
        Creates and returns a list of validation once datasets (only used for best epoch of training) from the given configuration.

        Returns:
            List[TrialDataset] or None: A list of TrialDataset instances, or None if not configured.
        """
        if self.validation_once_datasets is not None:
            return self.validation_once_datasets
        
        if self.config_validation_dataset is None:
            logger.info("No validation configured.")
            return None
        
        once_dataset = self.config_validation_dataset.get('once_dataset')
        if once_dataset is None:
            logger.info("No validation once dataset configured.")
            return None

        # Load once_dataset value as trial_dataset temporarily
        config_copy = copy.deepcopy(self.config_validation_dataset)
        config_copy['trial_dataset'] = config_copy.pop('once_dataset')

        self.validation_once_datasets = self._create_evaluation_datasets(config_copy)
        return self.validation_once_datasets
    

    def create_test_datasets(self):
        """
        Creates and returns a list of test datasets from the given configuration.

        Returns:
            List[TrialDataset] or None: A list of TrialDataset instances, or None if not configured.
        """
        if self.test_datasets is not None:
            return self.test_datasets
        
        if self.config_test_dataset is None:
            logger.info("No test dataset configured.")
            return None
        
        self.test_datasets = self._create_evaluation_datasets(self.config_test_dataset)
        return self.test_datasets
    

    def create_speaker_embedder(self):
        """
        Creates and returns the speaker embedding model.

        Returns:
            SpeakerEmbedderModel: The initialised speaker embedder.
        """
        if self.speaker_embedder is not None:
            return self.speaker_embedder

        self.speaker_embedder = SpeakerEmbedderModel(self.config_backend, self.config_frontend, self.config_classifier, self.model_checkpoint_path, self.device)
        return self.speaker_embedder


    def create_embedding_projection(self):
        """
        Creates and returns the embedding projection layer.

        Returns:
            nn.Module: The initialised embedding projection.
        """
        if self.embedding_projection is not None:
            return self.embedding_projection
        
        if self.speaker_embedder is None:
            self.speaker_embedder = self.create_speaker_embedder()
        
        emb_dim = self.speaker_embedder.get_backend_embedding_dim()

        if self.train_dataset is None:
            self.create_train_dataloader()
        num_classes = self.train_dataset.get_num_classes()

        if self.embedding_loss is None:
            self.embedding_loss = self.create_embedding_loss()

        use_genuine_labels = False if self.config_train_dataset.get('speaker_label_encoder') != "SpkLabelEncoder" else True
        self.config_embedding_projection.get('projection_args', {})['use_genuine_labels'] = use_genuine_labels

        self.embedding_projection = model_getter.get_embedding_projection(self.config_embedding_projection.get('projection_name'))(emb_dim, num_classes, criterion=self.embedding_loss, device=self.device, **self.config_embedding_projection.get('projection_args', {}))
        return self.embedding_projection
    

    def create_embedding_loss(self):
        """
        Creates and returns the embedding loss function.

        Returns:
            loss: initialised loss function for embedding training.
        """
        if self.embedding_loss is not None:
            return self.embedding_loss
        
        loss_config = self.config_embedding_projection.get('loss', {})
        self.embedding_loss = model_getter.get_loss_function(loss_config.get('loss_name', "CrossEntropyLoss"))(**loss_config.get('loss_args', {}))
        logger.info(f"Initialised embedding loss: {self.embedding_loss.__class__.__name__}")
        return self.embedding_loss
    

    def create_margin_scheduler(self, embedding_projection=None):
        """
        Creates and returns a margin scheduler to control the margin in loss functions dynamically.

        Args:
            embedding_projection (optional): Embedding projection module to use. If None, uses the internally created one.

        Returns:
            margin_scheduler: initialised margin scheduler object or None if not configured.
        """
        if self.margin_scheduler is not None:
            return self.margin_scheduler

        if self.config_margin_scheduler is None:
            logger.info("No margin scheduler configured.")
            return None
        
        if embedding_projection is None:
            if self.embedding_projection is None:
                self.embedding_projection = self.create_embedding_projection()
            embedding_projection = self.embedding_projection
        
        if self.steps_per_epoch is None:
            self.steps_per_epoch = self._get_steps_per_epoch()
        
        self.margin_scheduler = model_getter.get_margin_scheduler(self.config_margin_scheduler.get('scheduler_name'))(embedding_projection, steps_per_epoch=self.steps_per_epoch, **self.config_margin_scheduler.get('scheduler_args', {}))
        logger.info(f"Changed initial margin of {embedding_projection.__class__.__name__} to {self.embedding_projection.margin}.")
        return self.margin_scheduler
    

    def create_classifier_loss(self):
        """
        Creates and returns the classifier loss function.

        Returns:
            loss: initialised loss function for classification training or None if no classifier configured.
        """
        if self.classifier_loss is not None:
            return self.classifier_loss
        
        if self.config_classifier is None:
            logger.info("No classifier loss configured.")
            return None
        
        loss_config = self.config_classifier.get('loss', {})
        self.classifier_loss = model_getter.get_loss_function(loss_config.get('loss_name', "BCEWithLogitsLoss"))(**loss_config.get('loss_args', {}))
        logger.info(f"Initialised classifier loss: {self.classifier_loss.__class__.__name__}")
        return self.classifier_loss
    

    def create_optimiser(self):
        """
        Creates and returns the optimiser for training.

        Returns:
            optimiser: initialised optimiser.
        """
        if self.optimiser is not None:
            return self.optimiser
        
        if self.speaker_embedder is None:
            self.speaker_embedder = self.create_speaker_embedder()
        if self.embedding_projection is None:
            self.embedding_projection = self.create_embedding_projection()
        if self.loss_balancer is None:
            self.loss_balancer = self.create_loss_balancer()

        backend_params = list(chain(self.speaker_embedder.get_trainable_parameters_backend(), self.embedding_projection.parameters(), self.loss_balancer.parameters()))
        ssl_params = list(self.speaker_embedder.get_trainable_parameters_ssl())
        
        optimiser_args = self.config_optimiser.get("optimiser_args", {})
        backend_cfg = optimiser_args.get("backend", {}) or {}
        ssl_cfg = optimiser_args.get("ssl", {}) or {}
        other_args = optimiser_args.get("other_args", {}) or {}

        param_groups = []

        # Backend group
        backend_group = {**backend_cfg, "params": backend_params, "name": "backend"}
        param_groups.append(backend_group)

        # SSL group
        if ssl_params:
            ssl_group = {**ssl_cfg, "params": ssl_params, "name": "ssl"}
            param_groups.append(ssl_group)

        # Create optimiser
        self.optimiser = model_getter.get_optimiser(self.config_optimiser.get("optimiser_name", "Adam"))(param_groups, **other_args)
        
        # Logging
        log_msg = (f"Initialised optimiser: {self.optimiser.__class__.__name__} (backend: {backend_cfg}")
        if ssl_params:
            log_msg += f", ssl: {ssl_cfg}"
        else:
            log_msg += ", no separate SSL group"
        log_msg += ")"
        logger.info(log_msg)

        return self.optimiser


    def create_lr_scheduler(self, optimiser=None):
        """
        Creates and returns the learning rate scheduler.

        Args:
            optimiser (optional): Optimiser instance to apply scheduler to. If None, uses internally created one.

        Returns:
            LambdaLR: Initialised learning rate scheduler or None if not configured.
        """
        if self.lr_scheduler is not None:
            return self.lr_scheduler

        if self.config_lr_scheduler is None:
            logger.info("No learning rate scheduler configured.")
            return None
        
        if optimiser is None:
            if self.optimiser is None:
                self.optimiser = self.create_optimiser()
            optimiser = self.optimiser

        if self.train_dataloader is None:
            self.train_dataloader = self.create_train_dataloader()
        
        if self.steps_per_epoch is None:
            self.steps_per_epoch = self._get_steps_per_epoch()

        total_epochs = self.config_train.get('num_epochs')

        # Base learning scheduler for backend
        base_scheduler = model_getter.get_lr_scheduler(self.config_lr_scheduler.get('scheduler_name'))(optimiser, steps_per_epoch=self.steps_per_epoch, total_epochs=total_epochs, **self.config_lr_scheduler.get("scheduler_args", {}))

        # Get SSL LR from config
        optimiser_args = self.config_optimiser.get("optimiser_args", {})
        ssl_cfg = optimiser_args.get("ssl", {}) or {}
        ssl_lr = ssl_cfg.get("lr", None)

        # Wrap scheduler so it doesn't touch SSL params
        self.lr_scheduler = SchedulerWithFixedLR(base_scheduler, optimiser, ssl_group_name="ssl", ssl_lr=ssl_lr)

        return self.lr_scheduler
    

    def create_loss_balancer(self):
        """
        Creates and returns the loss balancer.

        Returns:
            nn.Module: Initialised loss balancer or None if not configured.
        """
        if self.loss_balancer is not None:
            return self.loss_balancer
        
        use_deepfake = False if self.config_train_dataset.get('speaker_label_encoder') != "SpkLabelEncoder" else True
        use_classifier = False if self.config_classifier is None else True
        
        self.loss_balancer = model_getter.get_loss_balancer(self.config_loss_balancer.get('loss_balancer_name'))(use_deepfake, use_classifier, **self.config_loss_balancer.get('loss_balancer_args', {}))
        return self.loss_balancer

    
    def _get_steps_per_epoch(self):
        """
        Returns:
            int: Number of steps per epoch.
        """
        if self.steps_per_epoch is not None:
            return self.steps_per_epoch

        if self.train_dataloader is None:
            self.train_dataloader = self.create_train_dataloader()
        
        accumuluation_interval = self.config_train.get('accumulation_interval', 1)
        
        self.steps_per_epoch = get_num_accumulation_steps_per_epoch(self.train_dataloader, accumuluation_interval)
        return self.steps_per_epoch
