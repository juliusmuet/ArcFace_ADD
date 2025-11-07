# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from transformers import get_cosine_schedule_with_warmup as _setup_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR
import logging
import torch
from torch import nn

logger = logging.getLogger(__name__)


class SchedulerWithFixedLR(nn.Module):
    """
    Wrapper around a learning rate scheduler that allows keeping one specific
    parameter group's learning rate fixed while scheduling others.

    This is useful in cases such as self-supervised learning (SSL), where
    you may want to keep a specific parameter group (e.g., SSL head) at a
    fixed learning rate while allowing other groups to be updated according
    to a scheduler.
    """

    def __init__(self, scheduler, optimiser, ssl_group_name="ssl", ssl_lr=None):
        """
        Initialize the SchedulerWithFixedLR.

        Args:
            scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler to wrap.
            optimiser (torch.optim.Optimizer): The optimizer associated for the scheduler.
            ssl_group_name: Name of the param group to keep fixed (default: "ssl").  
            ssl_lr (float or None): Fixed learning rate for the selected parameter group. If None, the LR is not overridden (default: None).
        """
        self.scheduler = scheduler
        self.optimiser = optimiser
        self.ssl_lr = ssl_lr

        # Find the SSL group index by name
        self.ssl_group_idx = None
        for idx, group in enumerate(self.optimiser.param_groups):
            if group.get('name') == ssl_group_name:
                self.ssl_group_idx = idx
                break
        
        if self.ssl_group_idx is None:
            logger.warning(f"No parameter group named '{ssl_group_name}' found.")
        else:
            if self.ssl_lr is not None:
                self.optimiser.param_groups[self.ssl_group_idx]['lr'] = self.ssl_lr

    def step(self, iter=None):
        """
        Perform a scheduler step and then restore the fixed learning rate
        for the specified parameter group (if applicable).

        This ensures that even if the scheduler modifies the learning rate
        of all parameter groups, the selected group will retain the fixed
        SSL learning rate.

        Args:
            iter (int): The current iteration number (default: None).
        """
        # Update learning rates using the underlying scheduler
        if iter is not None:
            self.scheduler.step(iter)
        else:
            self.scheduler.step()

        # Restore fixed LR for SSL group
        if self.ssl_group_idx is not None and self.ssl_lr is not None:
            self.optimiser.param_groups[self.ssl_group_idx]['lr'] = self.ssl_lr


def _setup_exponential_schedule_with_warmup(optimiser, num_warmup_steps, gamma):
    """
    Creates a learning rate scheduler with a linear warmup phase followed by exponential decay.

    Args:
        optimiser (Optimizer): The optimiser for which to schedule the learning rate.
        num_warmup_steps (int): Number of training steps for linear warmup.
        gamma (float): Multiplicative factor for exponential decay after warmup per training step.

    Returns:
        LambdaLR: A PyTorch LambdaLR scheduler with the specified warmup and exponential decay behavior.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            return gamma ** (current_step - num_warmup_steps)
    
    return LambdaLR(optimiser, lr_lambda)


def _setup_exponential_schedule_with_warmup_bounded(optimiser, initial_lr, target_lr, num_warmup_steps, num_training_steps):
    """
    Creates a learning rate scheduler that warms up linearly, then decays exponentially
    from `initial_lr` to `target_lr` over the remaining training steps.

    Args:
        optimiser (Optimizer): The optimiser for which to schedule the learning rate.
        initial_lr (float): The starting learning rate after warmup.
        target_lr (float): The learning rate at the end of training.
        num_warmup_steps (int): Number of training steps for linear warmup.
        num_training_steps (int): Total number of training steps.

    Returns:
        LambdaLR: A PyTorch LambdaLR scheduler with bounded exponential decay after warmup.
    """
    decay_steps = max(1, num_training_steps - num_warmup_steps) # Steps after warmup

    # Compute gamma so that lr decays from initial_lr to target_lr over decay_steps
    gamma = (target_lr / initial_lr) ** (1 / decay_steps)

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            decay_step = current_step - num_warmup_steps
            return gamma ** decay_step

    return LambdaLR(optimiser, lr_lambda)


def ExponentialDecrease(optimiser, steps_per_epoch, total_epochs, warmup_epoch, target_lr=None, gamma=0.999):
    """
    Creates an exponential learning rate scheduler with a warmup phase.
    If `initial_lr` and `target_lr` are provided, uses a bounded exponential schedule.
    Otherwise, uses unbounded exponential decay with the provided `gamma`.

    Args:
        optimiser (Optimizer): The optimiser for which to schedule the learning rate.
        steps_per_epoch (int): Number of training steps in one epoch.
        total_epochs (int): Number of total training epochs.
        warmup_epoch (int): Epoch to end warmup (start indexing from 0).
                            Set to 0 to disable warmup.
        target_lr (float or None): Target learning rate at end of training (default: None).
                                   If provided, uses a bounded exponential schedule.
        gamma (float): Decay rate used if `initial_lr` and `target_lr` are None (default: 0.999).

    Returns:
        LambdaLR: The configured learning rate scheduler.
    
    Notes:
        - At step steps_per_epoch * warmup_epoch - 1 the learning rate becomes initial_lr.
        - If warmup_epoch is 0, then the learning rate becomes initial_lr at step 0.
        - If initial_lr is None, then the optimizers initial learning rate is used.
        - At step steps_per_epoch * total_epochs - 1 the learning rate becomes target_lr.
    """
    num_training_steps = steps_per_epoch * total_epochs - 1
    num_warmup_steps = steps_per_epoch * warmup_epoch - 1
    if warmup_epoch == 0:
        num_warmup_steps = 0

    if target_lr is not None:
        initial_lr = optimiser.param_groups[0]['lr']
        logger.info(
            f"Initialised Bounded Exponential Learning Rate Scheduler with parameters: \n"
            f"(initial_lr={initial_lr}, "
            f"target_lr={target_lr},"
            f"num_warmup_steps={num_warmup_steps}, "
            f"num_training_steps={num_training_steps})"
        )
        return _setup_exponential_schedule_with_warmup_bounded(optimiser, initial_lr, target_lr, num_warmup_steps, num_training_steps)
    else:
        logger.info(
            f"Initialised Exponential Learning Rate Scheduler with parameters: \n"
            f"(gamma={gamma}, "
            f"num_warmup_steps={num_warmup_steps})"
        )
        return _setup_exponential_schedule_with_warmup(optimiser, num_warmup_steps, gamma)


def CosineDecrease(optimiser, steps_per_epoch, total_epochs, warmup_epoch):
    """
    Creates a cosine annealing learning rate scheduler with a warmup phase.

    Args:
        optimiser (Optimizer): The optimiser for which to schedule the learning rate.
        steps_per_epoch (int): Number of training steps in one epoch.
        total_epochs (int): Number of total training epochs.
        warmup_epoch (int): Epoch to end warmup (start indexing from 0).
                            Set to 0 to disable warmup.

    Returns:
        LambdaLR: A cosine learning rate scheduler with warmup.
    
    Notes:
        - At step steps_per_epoch * warmup_epoch - 1 the learning rate becomes initial_lr.
        - If warmup_epoch is 0, then the learning rate becomes initial_lr at step 0.
        - If initial_lr is None, then the optimizers initial learning rate is used.
        - At step steps_per_epoch * total_epochs - 1 the learning rate becomes 0.
    """
    num_training_steps = steps_per_epoch * total_epochs - 1
    num_warmup_steps = steps_per_epoch * warmup_epoch - 1
    if warmup_epoch == 0:
        num_warmup_steps = 0

    logger.info(
            f"Initialised Cosine Learning Rate Scheduler with parameters: \n"
            f"(num_warmup_steps={num_warmup_steps}, "
            f"num_training_steps={num_training_steps})"
        )
    return _setup_cosine_schedule_with_warmup(optimiser, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)