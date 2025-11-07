# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import sys
import random
from datetime import timedelta
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def set_seed(seed=42):
    """
    Set the random seed for reproducibility across NumPy, Python's random module, and PyTorch (CPU and GPU).
    
    Parameters:
        seed (int): The seed value to use for all random number generators (default: 42).

    This function ensures that experiments are reproducible by setting the same seed for:
    - NumPy
    - Python's built-in random module
    - PyTorch on CPU and GPU (including all CUDA devices)

    Note:
        - Uncomment `torch.backends.cudnn.deterministic = True` for full reproducibility, 
          but it may slow down training.
        - `torch.backends.cudnn.benchmark = True` can improve performance, but might reduce determinism.
    """
    
    np.random.seed(seed)    # Set seed for NumPy random number generator

    random.seed(seed)   # Set seed for Python's built-in random module

    torch.manual_seed(seed) # Set seed for PyTorch CPU operations

    torch.cuda.manual_seed(seed)    # Set seed for PyTorch CUDA (GPU) operations on current device

    torch.cuda.manual_seed_all(seed)    # Set seed for all CUDA devices (when using multiple GPUs)

    # Uncomment the line below to make cuDNN operations deterministic (slower but fully reproducible)
    # torch.backends.cudnn.deterministic = True

    # Enable cuDNN benchmarking to select optimal algorithms for the hardware (faster but less deterministic)
    torch.backends.cudnn.benchmark = True

    logger.info(f"Set seed to {seed}")


def get_num_accumulation_steps_per_epoch(dataloader, accumulation_interval):
    """
    Calculate the number of accumulation steps per epoch.

    Parameters:
        dataloader (torch.utils.data.DataLoader): The PyTorch DataLoader for the dataset.
        accumulation_interval (int): The interval for accumulating gradients.

    Returns:
        int: The number of accumulation steps per epoch.
    """
    num_batches = len(dataloader)
    num_accumulation_steps_per_epoch = int(np.ceil(num_batches / accumulation_interval))
    return num_accumulation_steps_per_epoch


def setup_logging(logfile):
    """
    Setup logging to both a file and the console.

    Parameters:
        logfile (str): The path to the log file.
    """
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler()
        ]
    )

    # Global uncaught exception logging
    def handle_uncaught(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.getLogger().critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_uncaught


def get_elapsed_remaining_time_from_progress_bar(progress_bar):
    """
    Extracts and formats the elapsed and estimated remaining time from a tqdm progress bar.

    Args:
        progress_bar (tqdm.tqdm): A tqdm progress bar instance from which to extract timing info.

    Returns:
        tuple[str, str]: A tuple containing:
            - elapsed_str (str): The elapsed time since the progress bar started, formatted as 'HH:MM:SS'.
            - remaining_str (str): The estimated remaining time until completion, formatted as 'HH:MM:SS'.
    """
    elapsed_secs = progress_bar.format_dict.get('elapsed', 0)
    rate = progress_bar.format_dict.get('rate', None)
    total = progress_bar.format_dict.get('total', None)
    n = progress_bar.format_dict.get('n', None)
    if rate and total and n:
        remaining_secs = (total - n) / rate
    else:
        remaining_secs = 0
    elapsed_str = str(timedelta(seconds=int(elapsed_secs)))
    remaining_str = str(timedelta(seconds=int(remaining_secs)))
    return elapsed_str, remaining_str
