# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import logging
import backends_wespeaker.ecapa_tdnn as ecapa_tdnn
import backends_wespeaker.resnet as resnet
import backends_wespeaker.campplus as campplus
import backends_wespeaker.gemini_dfresnet as gemini
import backends_wespeaker.samresnet as samresnet
import backends_3dspeaker.ecapa_tdnn.ECAPA_TDNN as ecapa_tdnn_3dspeaker
import backends_3dspeaker.campplus.DTDNN as campplus_3dspeaker
import backends_3dspeaker.eres2net.ERes2Net as ERes2Net
import classifiers.length_classifier as length_classifier
import losses.arcface as arcface
import losses.margin_scheduler as margin_schedulers
import train.lr_scheduler as lr_schedulers
import datasets.speaker_labels as encoders
import datasets.samplers as batch_sampler
import train.loss_balancer as loss_balancer
import frontends.adapters as adapters

logger = logging.getLogger(__name__)


"""
Module for dynamically resolving and loading classes by name prefixes across different components 
of a speaker recognition pipeline, including backends, classifiers, loss functions, optimisers, 
learning rate schedulers and label encoders.
"""

def get_class_by_prefix(name, prefix_module_mapping, item_type="Item"):
    """
    Fetches a class from the appropriate module based on the provided name prefix.

    Args:
        name (str): The name of the class to load.
        prefix_module_mapping (dict): A dictionary where keys are prefixes and values are modules.
        item_type (str): A string for identifying the item type in error messages (default: "item").

    Returns:
        class: The resolved class from the module.

    Raises:
        SystemExit: If the name does not match any known prefix.
    """
    for prefix, module in prefix_module_mapping.items():
        if name.startswith(prefix):
            return getattr(module, name)
    logger.info(f"{item_type} '{name}' not found!")
    exit(1)


# Prefix to module mappings
backend_model_prefixes = {
    "ERes2Net": ERes2Net,
    "ECAPA_TDNN_3DSpeaker": ecapa_tdnn_3dspeaker,
    "CAMPPlus_3DSpeaker": campplus_3dspeaker,
    "ECAPA_TDNN": ecapa_tdnn,
    "ResNet": resnet,
    "CAMPPlus": campplus,
    "Gemini": gemini,
    "SimAM_ResNet": samresnet
}

classifier_prefixes = {
    "Length": length_classifier,
}

embedding_projection_prefixes = {
    "ArcFace": arcface,
}

margin_scheduler_prefixes = {
    "ArcFace": margin_schedulers,
}

optimiser_prefixes = {
    "Adam": torch.optim,
    "SGD": torch.optim,
}

lr_scheduler_prefixes = {
    "Exponential": lr_schedulers,
    "Cosine": lr_schedulers,
}

loss_function_prefixes = {
    "CrossEntropy": torch.nn,
    "BCEWith": torch.nn,
}

encoder_prefixes = {
    "Spk": encoders,
    "Grouped": encoders,
    "Alternating": encoders,
    "DeepfakeUnified": encoders,
}

batch_sampler_prefixes = {
    "SpeakerGenuineFakeVocoderBalanced": batch_sampler,
    "SpeakerGenuineFakeBalanced": batch_sampler,
    "SpeakerBalanced": batch_sampler,
    "GenuineFakeBalanced": batch_sampler
}

loss_balancer_prefixes = {
    "WeightedSum" : loss_balancer,
    "UncertaintyWeighting" : loss_balancer
}

adapter_prefixes = {
    "SimpleAdapter": adapters,
    "ResidualAdapter": adapters
}


def get_backend_model(name):
    """
    Retrieves a backend model class based on its name prefix.

    Supported model names and their corresponding modules:
    - ERes2Net
    - ECAPA_TDNN_3DSpeaker
    - CAMPPlus_3DSpeaker
    - ECAPA_TDNN_GLOB_c512
    - ECAPA_TDNN_GLOB_c1024
    - ResNet34
    - ResNet152
    - ResNet221
    - ResNet293
    - CAMPPlus
    - Gemini_DF_ResNet114
    - SimAM_ResNet34_ASP
    - SimAM_ResNet100_ASP

    Args:
        name (str): The name of the backend model class.

    Returns:
        class: The backend model class.
    """
    return get_class_by_prefix(name, backend_model_prefixes, "Backend Model")


def get_classifier_model(name):
    """
    Retrieves a classifier model class based on its name prefix.

    Supported classifier names:
    - LengthBasedClassifier

    Args:
        name (str): The name of the classifier model.

    Returns:
        class: The classifier model class.
    """
    return get_class_by_prefix(name, classifier_prefixes, "Classifier Model")


def get_embedding_projection(name):
    """
    Retrieves an embedding projection module based on its name prefix.

    Supported projection names:
    - ArcFace

    Args:
        name (str): The name of the embedding projection.

    Returns:
        class: The embedding projection class.
    """
    return get_class_by_prefix(name, embedding_projection_prefixes, "Embedding Projection")


def get_margin_scheduler(name):
    """
    Retrieves a margin scheduler based on its name prefix.

    Supported scheduler names:
    - ArcFaceMarginScheduler

    Args:
        name (str): The name of the margin scheduler.

    Returns:
        class: The margin scheduler class.
    """
    return get_class_by_prefix(name, margin_scheduler_prefixes, "Margin Scheduler")


def get_optimiser(name):
    """
    Retrieves an optimiser class from torch.optim based on its name prefix.

    Supported optimiser names:
    - Adam
    - SGD

    Args:
        name (str): The name of the optimiser.

    Returns:
        class: The optimiser class.
    """
    return get_class_by_prefix(name, optimiser_prefixes, "Optimiser")


def get_lr_scheduler(name):
    """
    Retrieves a learning rate scheduler from a scheduler module based on its name prefix.

    Supported scheduler names:
    - ExponentialDecrease
    - CosineDecrease

    Args:
        name (str): The name of the learning rate scheduler.

    Returns:
        class: The learning rate scheduler class.
    """
    return get_class_by_prefix(name, lr_scheduler_prefixes, "Learning Rate Scheduler")


def get_loss_function(name):
    """
    Retrieves a loss function class from torch.nn based on its name prefix.

    Supported loss functions:
    - CrossEntropyLoss
    - BCEWithLogitsLoss

    Args:
        name (str): The name of the loss function.

    Returns:
        class: The loss function class.
    """
    return get_class_by_prefix(name, loss_function_prefixes, "Loss Function")


def get_speaker_label_encoder(name):
    """
    Retrieves a speaker label encoder class based on its name prefix.

    Supported encoder names:
    - SpkLabelEncoder
    - GroupedGenuineDeepfakePairEncoder
    - AlternatingGenuineDeepfakePairEncoder
    - DeepfakeUnifiedEncoder

    Args:
        name (str): The name of the speaker label encoder.

    Returns:
        class: The encoder class.
    """
    return get_class_by_prefix(name, encoder_prefixes, "Speaker Label Encoder")


def get_batch_sampler(name):
    """
    Retrieves a batch sampler class based on its name prefix.

    Supported encoder names:
    - SpeakerBalancedBatchSampler

    Args:
        name (str): The name of the batch sampler.

    Returns:
        class: The batch sampler.
    """ 
    return get_class_by_prefix(name, batch_sampler_prefixes, "Batch Sampler")


def get_loss_balancer(name):
    """
    Retrieves a loss balancer class based on its name prefix.

    Supported encoder names:
    - WeightedSum
    - UncertaintyWeighting

    Args:
        name (str): The name of the loss balancer.

    Returns:
        class: The loss balancer.
    """ 
    return get_class_by_prefix(name, loss_balancer_prefixes, "Loss Balancer")  


def get_adapter(name):
    """
    Retrieves a adapter class based on its name prefix.

    Supported encoder names:
    - SimpleAdapter
    - ResidualAdapter

    Args:
        name (str): The name of the adapter.

    Returns:
        class: The adapter.
    """ 
    return get_class_by_prefix(name, adapter_prefixes, "Adapter")  