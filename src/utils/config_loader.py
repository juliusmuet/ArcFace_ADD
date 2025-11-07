# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import yaml


def load_config_file(config_path):
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_model_checkpoint(config):
    return config.get('model_checkpoint', None)

def get_train_config(config):
    return config.get('train_args', {})

def get_loss_balancer_config(config):
    return config.get('loss_balancer', {'loss_balancer_name': "WeightedSum"})

def get_train_dataset_config(config):
    return config.get('train_dataset', {})

def get_validation_config(config):
    return config.get('validation_args', {})

def get_validation_dataset_config(config):
    return config.get('validation_dataset', None)

def get_test_config(config):
    return config.get('test_args', {})

def get_test_dataset_config(config):
    return config.get('test_dataset', None)

def get_train_dataloader_config(config):
    return config.get('train_dataloader', {})

def get_preprocessing_config(config):
    return config.get('preprocessing', {})

def get_frontend_config(config):
    return config.get('frontend', {'model_name': "wavlm",
                                   'model_args': {}})

def get_backend_config(config):
    return config.get('backend', {'model_name': "SimAM_ResNet34_ASP",
                                  'model_args': {},
                                  'checkpoint_path': r"Pretrained_Models_WeSpeaker\voxblink2_samresnet34_ft\avg_model.pt"})

def get_embedding_projection_config(config):
    return config.get('embedding_projection', {'projection_name': "ArcFace",
                                               'projections_args': {}, 
                                               'loss': {'loss_name': "CrossEntropyLoss", 'loss_args': {}}})

def get_margin_scheduler_configs(config):
    return config.get('margin_scheduler', None)

def get_classifier_config(config):
    return config.get('classifier', None)

def get_optimiser_config(config):
    return config.get('optimiser', {'optimiser_name': "Adam",
                                    'optimiser_args': {}})

def get_lr_scheduler_config(config):
    return config.get('lr_scheduler', None)