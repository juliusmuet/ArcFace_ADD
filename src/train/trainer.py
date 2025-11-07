# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import json
from pathlib import Path
from typing import Optional, Any, List
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from utils.utils import get_num_accumulation_steps_per_epoch
from datasets.dataset_evaluation import TrialDataset
from train.evaluation import evaluate_speaker_verification_spoofed, evaluate_speaker_verification
from preprocessing.wav_loader import WavLoader
from utils.utils import get_elapsed_remaining_time_from_progress_bar

logger = logging.getLogger(__name__)


class Trainer:
    """
    A class to handle the training loop for a PyTorch model,
    including an ArcFace projection head and optional direct model logits.

    Args:
        model: The main PyTorch model.
        embedding_projection: The embedding projection layer with loss (e.g. ArcFace).
        train_dataloader: DataLoader for the training data.
        optimiser: The optimiser for the model's trainable parameters.
        device: The device to train on (e.g., 'cuda').
        scaler: GradScaler for mixed precision training.
        lr_scheduler: Learning rate scheduler.
        margin_scheduler: Scheduler for the embedding projections's margin. Expected to have
                          `get_margin()` and `step()` methods (behavior depends on scheduler design).
        criterion_classifier: Loss function for the model's optional direct logits (default: None).
        loss_balancer: Loss balancer module (default: None).
        num_epochs: Total number of epochs to train for (default: 25).
        log_interval: How often to log training progress (in batches) (default: 100).
        accumulation_interval: Number of steps to accumulate gradients before an optimizer step (default: 1).
        save_epoch_interval: How often to save the model (in epochs) (default: 1).
        lm_finetune: Whether large-margin fine-tuning is applied (default: False).
    """
    def __init__(self,
                 model: nn.Module,
                 embedding_projection: nn.Module,
                 train_dataloader: DataLoader,
                 optimiser: Optimizer,
                 device: str,
                 scaler: torch.GradScaler,
                 lr_scheduler: Any = None,
                 margin_scheduler: Any = None,
                 criterion_classifier: Optional[nn.Module] = None,
                 loss_balancer: nn.Module = None,
                 num_epochs: int = 25,
                 log_interval: int = 100,
                 accumulation_interval: int = 1,
                 save_epoch_interval: int = 1,
                 lm_finetune: bool = False
                 ):
        self.model = model.to(device)
        self.embedding_projection = embedding_projection.to(device)
        self.train_dataloader = train_dataloader
        self.optimiser = optimiser
        self.device = device
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler
        self.margin_scheduler = margin_scheduler
        self.criterion_classifier = criterion_classifier
        self.loss_balancer = loss_balancer
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.accumulation_interval = max(1, accumulation_interval)
        self.save_epoch_interval = save_epoch_interval
        self.lm_finetune = lm_finetune
        self.steps_per_epoch = get_num_accumulation_steps_per_epoch(self.train_dataloader, self.accumulation_interval)

        logger.info(f"Initialised Trainer with parameters:\n{self}")
    

    def __str__(self):
        return (
            f"(num_epochs: {self.num_epochs}, "
            f"log_interval: {self.log_interval}, "
            f"accumulation_interval: {self.accumulation_interval}, "
            f"save_epoch_interval: {self.save_epoch_interval}, "
            f"lm_finetune: {self.lm_finetune}, "
            f"steps_per_epoch: {self.steps_per_epoch})"
        )


    def train_epoch(self, epoch_num: int) -> float:
        """
        Trains the model for one epoch.

        Args:
            dataloader: DataLoader for the training data.
                        Expected to yield (model_input, speaker_label, is_genuine_flag).
            epoch_num: The current epoch number (start indexing from 1).

        Returns:
            The average total loss for the epoch.
        """
        # Set model and embedding projection to train mode
        self.model.train()
        self.embedding_projection.train()

        # Define variables
        total_loss = 0.0
        total_embedding_loss = 0.0
        total_classifier_loss = 0.0
        num_batches = len(self.train_dataloader)

        # Set current steps in schedulers if defined (needed if start epoch in train() is not 1)
        current_step = (epoch_num - 1) * self.steps_per_epoch
        if self.lr_scheduler:
            self.lr_scheduler.step(current_step)
        if self.margin_scheduler:
            self.margin_scheduler.step(current_step)

        # Wrap the data loader with tqdm for a progress bar
        progress_bar = tqdm(self.train_dataloader, desc="Epoch Training", leave=False)

        # Batch training loop
        for batch_idx, batch_data in enumerate(progress_bar):
            # Ensure batch_data has the correct number of elements
            if len(batch_data) != 3:
                raise ValueError(f"DataLoader expected to yield 3 items (inputs, speaker_labels, genuine_flags), but got {len(batch_data)} items.")
            
            # Get data and put on device
            inputs, speaker_labels, genuine_flags = batch_data
            inputs = inputs.to(self.device)
            speaker_labels = speaker_labels.to(self.device)
            genuine_flags = genuine_flags.to(self.device).float()

            # Speeds up training by using mixed precision
            with torch.amp.autocast(self.device):
                # Get model outputs
                outputs = self.model(inputs)
                embedding, classifier_logits = outputs

                # Calculate embedding projection loss
                loss_emb_genuine, loss_emb_deepfake = self.embedding_projection(embedding, speaker_labels, genuine_flags)

                # Calculate classifier loss
                loss_classifier = self.criterion_classifier(classifier_logits, genuine_flags) if classifier_logits is not None else None

                # Compute weighted total loss
                batch_loss = self.loss_balancer(loss_emb_genuine, loss_emb_deepfake, loss_classifier)
                batch_loss = batch_loss / self.accumulation_interval   # Normalise loss for gradient accumulation

            self.scaler.scale(batch_loss).backward()    # Accumulate gradients

            # Un-normalise for logging
            total_loss += batch_loss.item() * self.accumulation_interval
            total_embedding_loss += loss_emb_genuine.item()
            if loss_emb_deepfake is not None:
                total_embedding_loss += loss_emb_deepfake.item()
            if loss_classifier is not None:
                total_classifier_loss += loss_classifier.item()

            # Update weights and schedulers after accumulation_steps or after last batch
            if (batch_idx + 1) % self.accumulation_interval == 0 or (batch_idx + 1) == num_batches:
                self.scaler.step(self.optimiser)    # Update weights
                self.scaler.update()    # Update scaler
                self.optimiser.zero_grad()  # Delete all gradients

                # Step the schedulers if defined
                current_step += 1
                if self.lr_scheduler:
                    self.lr_scheduler.step(current_step)
                if self.margin_scheduler:
                    self.margin_scheduler.step(current_step)

            # Update progress bar after every batch
            progress_bar.set_postfix({
                'Loss': f"{batch_loss.item()*self.accumulation_interval:.4f}",
                'Margin': f"{self.embedding_projection.get_margin():.6f}",
                'LR': ", ".join(f"{pg['lr']:.6e}" for pg in self.optimiser.param_groups)
            })
            
            # Log after self.log_interval many batches and after first batch
            if (batch_idx + 1) % self.log_interval == 0 or batch_idx == 0:
                loss_emb_proj = loss_emb_genuine.item() if loss_emb_deepfake is None else loss_emb_genuine.item() + loss_emb_deepfake.item()
                log_msg = (
                    f"Epoch: {epoch_num} [{batch_idx + 1}/{num_batches} ({100. * (batch_idx + 1) / num_batches:.0f}%)] | "
                    f"Batch Total Loss: {batch_loss.item()*self.accumulation_interval:.4f} | "
                    f"Batch Embedding Projection Loss: {loss_emb_proj:.4f}"
                )
                if loss_emb_deepfake is not None:
                    log_msg += f" (Genuine: {loss_emb_genuine.item():.4f}, Deepfake: {loss_emb_deepfake.item():.4f})"
                if classifier_logits is not None:
                    log_msg += f" | Batch Model Classifier Loss: {loss_classifier.item():.4f}"
                log_msg += f" | Margin: {self.embedding_projection.get_margin():.6f}"
                log_msg += f" | LR:" + ", ".join(f"{pg['lr']:.6e}" for pg in self.optimiser.param_groups)
                log_msg += f" | Loss Balancer: (genuine: {self.loss_balancer.get_genuine_parameter():.4f}), (deepfake: {self.loss_balancer.get_deepfake_parameter():.4f}), (classifier: {self.loss_balancer.get_classifier_parameter():.4f})"
                elapsed_str, remaining_str = get_elapsed_remaining_time_from_progress_bar(progress_bar)
                log_msg += f" | Elapsed: {elapsed_str} | ETA: {remaining_str}"
                logger.info(log_msg)

        # Log at end of epoch
        avg_total_loss = total_loss / num_batches
        avg_arcface_loss = total_embedding_loss / num_batches
        avg_model_logit_loss = total_classifier_loss / num_batches
        logger.info(f"--- Epoch {epoch_num} Summary ---")
        logger.info(f"Average Total Loss: {avg_total_loss:.4f}")
        logger.info(f"Average ArcFace Loss: {avg_arcface_loss:.4f}")
        if classifier_logits is not None:
            logger.info(f"Average Model Logit Loss: {avg_model_logit_loss:.4f}")
        logger.info(f"Current Margin at epoch end: {self.embedding_projection.get_margin():.6f}")
        logger.info(f"Current LR at epoch end: " + ", ".join(f"{pg['lr']:.6e}" for pg in self.optimiser.param_groups))
        logger.info(f"Current Loss Balancer at epoch end: (genuine: {self.loss_balancer.get_genuine_parameter():.4f}), (deepfake: {self.loss_balancer.get_deepfake_parameter():.4f}), (classifier: {self.loss_balancer.get_classifier_parameter():.4f})")
        logger.info("-------------------------")

        return avg_total_loss


    def train(self, start_epoch: int = 1, val_datasets: Optional[List[TrialDataset]] = None, val_config: dict = {}, checkpoint_save_dir: str = Path("./checkpoints"), checkpoint_load_path: Optional[str] = None, val_once_datasets: Optional[List[TrialDataset]] = None, base_path: str = None):
        """
        Main training loop for a specified number of epochs.
        If val_dataset is given, validation after each epoch is performed.

        Args:
            start_epoch : The epoch from which to start training (default: 1).
                         Start indexing from 1.
            val_datasets: Optional list of validation datasets (default: None).
            val_config: Configuration for validation (default: {})
            checkpoint_save_dir: Directory where checkpoints will be saved (default: './checkpoints').
            checkpoint_load_path: Path from where checkpoints will be loaded (default: None).
            val_once_datasets: Optional extra validation dataset, which is only used for the best epoch (default: None).
            base_path: Path to directory containing checkpoints (default: None).
        """
        if checkpoint_load_path:
            start_epoch = self.load_checkpoint(checkpoint_load_path, finetune=self.lm_finetune)

        # Store results per epoch
        best_epoch = None
        best_mode = None
        lowest_total_spf_eer = float('inf')
        all_results = {}
        
        # Store best spf_eer per dataset and mode
        best_per_dataset_mode = {}  # {(dataset_idx, mode): (epoch, spf_eer, result_entry)}

        progress_bar = tqdm(range(start_epoch, self.num_epochs+1), desc="Epochs")
        for epoch in progress_bar:
            # Log progress
            elapsed_str, remaining_str = get_elapsed_remaining_time_from_progress_bar(progress_bar)
            logger.info(f"Epoch {epoch} / {self.num_epochs} | Elapsed: {elapsed_str} | ETA: {remaining_str}")
            
            # Train
            total_loss = self.train_epoch(epoch)

            # Save checkpoints
            if epoch % self.save_epoch_interval == 0:
                self.save_checkpoint(epoch, checkpoint_save_dir)
            
            # Used when training plain automatic speaker verification
            """
            wavloader_eval = WavLoader()
            evaluate_speaker_verification(self.model, wavloader_eval, "/data/audio_data_ma/vox1_test_wav/vox1-O-clean.txt", "/data/audio_data_ma/vox1_test_wav/wav")
            """

            # Validate
            if val_datasets:
                visualise_save_path = base_path / f"{epoch}_visualisation"
                visualise_save_path.mkdir(parents=True, exist_ok=True)
                results = evaluate_speaker_verification_spoofed(model=self.model, datasets=val_datasets, device=self.device, visualise_save_path=visualise_save_path, **val_config)
                all_results[epoch] = results

                # Track best spf_eer per dataset and mode
                for dataset_idx, dataset_results in results.items():
                    for mode, metrics in dataset_results.items():
                        spf_eer = metrics.get("spf_eer", 1.0)
                        key = (dataset_idx, mode)
                        if key not in best_per_dataset_mode or spf_eer < best_per_dataset_mode[key][1]:
                            best_per_dataset_mode[key] = (epoch, spf_eer, metrics)

                # Find best mode for this epoch by summing spf_eer over datasets
                mode_spf_eer_sums = {}

                # Iterate over each dataset (key: "0", "1", ...)
                for dataset_key, dataset_results in results.items():
                    for mode_key, mode_metrics in dataset_results.items():
                        spf_eer = mode_metrics.get("spf_eer", 1.0)
                        if spf_eer is not None:
                            mode_spf_eer_sums.setdefault(mode_key, 0.0)
                            mode_spf_eer_sums[mode_key] += spf_eer

                # Check if any mode is best so far
                for mode, spf_eer_sum in mode_spf_eer_sums.items():
                    if spf_eer_sum < lowest_total_spf_eer:
                        lowest_total_spf_eer = spf_eer_sum
                        best_mode = mode
                        best_epoch = epoch
            
        logger.info(f"Training finished. Best validation epoch: {best_epoch}, best mode: {best_mode}, lowest total SPF_EER: {lowest_total_spf_eer:.5f}\n")

        # Log best per (dataset, mode)
        logger.info("Best per dataset and mode:")
        for (dataset_idx, mode), (epoch, spf_eer, metrics) in best_per_dataset_mode.items():
            logger.info(f"Dataset {dataset_idx} | Mode: {mode} | Best Epoch: {epoch}")
            metric_values_for_summary = []
            for k, v in metrics.items():
                if isinstance(v, float):
                    logger.info(f"  {k}: {v:.4f}")
                    if "threshold" not in k:
                        if "eer" in k:
                            metric_values_for_summary.append(f"{v*100:.4f}")
                        else:
                            metric_values_for_summary.append(f"{v:.4f}")
                else:
                    logger.info(f"  {k}: {v}")
            logger.info("\t".join(metric_values_for_summary) + "\n")

        # Test on extra validation dataset with best epoch and visualise
        if val_once_datasets:
            logger.info(f"Validating on validation once datasets with best epoch {best_epoch}...")
            self.load_checkpoint(base_path / f"checkpoint_epoch_{best_epoch}.pt", False)
            visualise_save_path = base_path / "once_val"
            visualise_save_path.mkdir(parents=True, exist_ok=True)
            results = evaluate_speaker_verification_spoofed(model=self.model, datasets=val_once_datasets, device=self.device, visualise_save_path=visualise_save_path, **val_config)
            
            # Log output per dataset and mode
            for dataset_idx, dataset_results in results.items():
                for mode, metrics in dataset_results.items():
                    metric_values_for_summary = []
                    logger.info(f"Dataset {dataset_idx} | Mode: {mode}")
                    for k, v in metrics.items():
                        if isinstance(v, float):
                            logger.info(f"  {k}: {v:.4f}")
                            if "threshold" not in k:
                                metric_values_for_summary.append(f"{v:.4f}")
                        else:
                            logger.info(f"  {k}: {v}")
                    logger.info("\t".join(metric_values_for_summary) + "\n")    # This line is used for easy copy into excel
            
            # Save results to file
            output_path = Path(visualise_save_path) / "results.txt"
            with open(output_path, "w") as f:
                f.write(json.dumps(results, indent=4))
            logger.info(f"Evaluation results (val once datasets) saved to {output_path}")
    

    def save_checkpoint(self, epoch: int, checkpoint_dir: str = Path('./checkpoints')):
        """
        Saves the current training state to a checkpoint file.

        Args:
            epoch (int): The current epoch number to include in the filename.
            checkpoint_dir (str): Directory to save the checkpoint in (default: './checkpoints').

        Saves:
            A .pt file containing:
                - epoch (int): Last completed epoch.
                - model_state_dict (dict): State dict of the model.
                - embedding_projection_state_dict (dict): State dict of the embedding projection (e.g., ArcFace).
                - optimizer_state_dict (dict): Optimizer state dict.
                - scaler_state_dict (dict): Mixed precision GradScaler state.
                - loss_balancer_dict (dict): Loss balancer state.
        
        Note:
            Scheduler states are *not* saved, assuming they are step-based and deterministically reset at epoch start.
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)  # Make sure checkpoint dir exists
        checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'embedding_projection_state_dict': self.embedding_projection.state_dict(),
            'optimizer_state_dict': self.optimiser.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'loss_balancer_dict': self.loss_balancer.state_dict()
        }
        # No need to save the schedulers as their step is set at the beginning of each epoch
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}\n")


    def load_checkpoint(self, checkpoint_path: str, finetune: bool = False):
        """
        Loads a previously saved training state from a checkpoint file.

        Args:
            checkpoint_path (str): Path to the checkpoint file (.pt).
            finetune (bool): If True, only loads model backbone weights for fine-tuning (default: False).

        Returns:
            int: The next epoch number to resume training from.

        Loads:
            - model_state_dict: Restores model parameters.
            - embedding_projection_state_dict: Restores ArcFace (or similar) embedding projection head.
            - optimizer_state_dict: Restores optimizer parameters and history.
            - scaler_state_dict: Restores mixed precision training state.
            - loss_balancer_dict: Restores loss balancer state.

        Note:
            LR and margin schedulers are not loaded here, as they are stepped manually
            and deterministically reinitialized in each epoch based on the current step.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load full model state
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # Comment this out if model is a pretrained ASV system trained with ArcFace as for audio deepfake detection the ArcFace dimensions are different
        self.embedding_projection.load_state_dict(checkpoint['embedding_projection_state_dict'], strict=False)

        if not finetune:
            self.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self.loss_balancer.load_state_dict(checkpoint['loss_balancer_dict'])
            logger.info(f"Checkpoint loaded from {checkpoint_path} with start epoch {checkpoint['epoch']+1}")
            return checkpoint['epoch'] + 1
        else:
            logger.info(f"Checkpoint loaded for fine-tuning from {checkpoint_path}. Starting from epoch 1.")
            return 1
