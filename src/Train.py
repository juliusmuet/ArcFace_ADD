# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
from datetime import datetime
from pathlib import Path
import logging
import torch
from utils.factory import Factory
from train.trainer import Trainer
from utils.utils import setup_logging
from utils.config_loader import load_config_file, get_model_checkpoint


def main():
    # Get input parameters
    parser = argparse.ArgumentParser(description="Train a speaker embedding model.")
    parser.add_argument("base_path", type=str, help="Base path to the config directory (must contain config.yml)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint filename (e.g., checkpoint_epoch_1.pt)")
    args = parser.parse_args()

    # Get paths
    config_path = Path(args.base_path) / "config.yml"
    checkpoint_load_path = Path(args.base_path) / args.checkpoint if args.checkpoint else None
    log_file = Path(args.base_path) / f"train_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

    checkpoint_path_from_file = get_model_checkpoint(load_config_file(config_path))
    checkpoint_load_path = Path(args.base_path) / checkpoint_path_from_file if checkpoint_path_from_file else checkpoint_load_path

    # Setup logger
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
    logger.info(f"Training with path {config_path} and checkpoint {checkpoint_load_path}")

    # Build all components for training from config
    factory = Factory(config_path)
    model = factory.create_speaker_embedder()
    embedding_projection = factory.create_embedding_projection()
    train_dataloder = factory.create_train_dataloader()
    val_datasets = factory.create_validation_datasets()
    val_once_datasets = factory.create_validation_once_datasets()
    val_config = factory.config_validation
    optimiser = factory.create_optimiser()
    device = factory.device
    scaler = torch.amp.GradScaler(device)
    lr_scheduler = factory.create_lr_scheduler()
    margin_scheduler = factory.create_margin_scheduler()
    criterion_classifier = factory.create_classifier_loss()
    loss_balancer = factory.create_loss_balancer()
    config_train = factory.config_train

    # Train
    trainer = Trainer(model, embedding_projection, train_dataloder, optimiser, device, scaler, lr_scheduler, margin_scheduler, criterion_classifier, loss_balancer, **config_train)
    trainer.train(val_datasets=val_datasets, val_config=val_config, checkpoint_save_dir=factory.config_directory, checkpoint_load_path=checkpoint_load_path, val_once_datasets=val_once_datasets, base_path=Path(args.base_path))

if __name__ == "__main__":
    main()