# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import math
import logging

logger = logging.getLogger(__name__)


class ArcFaceMarginScheduler:
    """
    A scheduler to adjust the margin of an ArcFace loss function during training.
    
    The scheduler gradually increases the margin from `initial_margin` to `target_margin`
    between `start_step` and `fix_step`, either linearly or exponentially. After `fix_step`,
    the margin remains fixed at `target_margin`.

    Attributes:
        arcface (ArcFace): The ArcFace object with an `update(margin=...)` method.
        steps_per_epoch (int): Number of training steps in one epoch.
        start_epoch (int): Epoch to start increasing the margin (start indexing from 0).
        fix_sepoch (int): Epoch at which the margin becomes fixed (start indexing from 0).
        initial_margin (float): Initial margin value (default: 0.0).
        target_margin (float): Final margin value after the increase phase (default: 0.2).
        increase_type (str): Type of increase ('exp' or 'linear') (default: 'exp').
    """

    def __init__(self, arcface, steps_per_epoch, start_epoch, fix_epoch, initial_margin=0.0, target_margin=0.2, increase_type='exp'):
        self.arcface = arcface
        self.start_step = start_epoch * steps_per_epoch - 1
        if start_epoch == 0:
            self.start_step = 0
        self.fix_step = fix_epoch * steps_per_epoch - 1
        if fix_epoch == 0:
            self.fix_step = 0
        self.initial_margin = initial_margin
        self.target_margin = target_margin
        self.increase_type = increase_type

        self.fixed = False
        self.current_step = 0
        self.effective_total_steps  = self.fix_step - self.start_step

        self._init_margin()

        logger.info(f"Initialised ArcFace Margin Scheduler with parameters:\n{self}")


    def __str__(self):
        return (
            f"ArcFaceMarginScheduler(start_step={self.start_step}, "
            f"fix_step={self.fix_step}, "
            f"initial_margin={self.initial_margin}, "
            f"target_margin={self.target_margin}, "
            f"increase_type='{self.increase_type}')"
        )


    def _init_margin(self):
        """
        Sets the initial margin on the ArcFace module.
        """
        if self.current_step >= self.fix_step:
            self.fixed = True
            self.arcface.update(margin=self.target_margin)
        elif self.current_step >= self.start_step:
            self.arcface.update(margin=self.initial_margin)


    def _get_increase_margin(self):
        """
        Calculates the current margin value based on the iteration,
        either using exponential or linear increase.

        Returns:
            float: The computed margin value.
        """
        progress_epochs = self.current_step - self.start_step

        if self.increase_type == 'exp':  # exponentially increase the margin
            ratio = 1.0 - math.exp((progress_epochs / self.effective_total_steps) * math.log(1e-3 / (1.0 + 1e-6)))
        else:  # linearly increase the margin
            ratio = progress_epochs / self.effective_total_steps 
        
        return self.initial_margin + (self.target_margin - self.initial_margin) * ratio


    def step(self, current_step=None):
        """
        Advances the scheduler by one iteration and updates the margin accordingly.

        Args:
            current_step (int): Current step number (default: None).
                                If None, internal step counter is used and incremented.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        if self.fixed:
            return

        if self.current_step >= self.fix_step:
            self.fixed = True
            self.arcface.update(margin=self.target_margin)
        elif self.current_step >= self.start_step:
            self.arcface.update(margin=self._get_increase_margin())
        

    def get_margin(self):
        """
        Retrieves the current margin value from the ArcFace module.

        Returns:
            float: Current margin value.
        """
        return self.arcface.get_margin()
