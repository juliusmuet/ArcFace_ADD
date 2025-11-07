# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
from datetime import datetime
from pathlib import Path
import logging
from utils.factory import Factory
from train.evaluation import evaluate_speaker_verification_spoofed
from utils.utils import setup_logging


def main():
    # Get path to config directory
    parser = argparse.ArgumentParser(description="Evaluate speaker verification model.")
    parser.add_argument("base_path", type=str, help="Base path to the config directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint file name")
    args = parser.parse_args()

    # Get paths
    config_path = Path(args.base_path) / "config.yml"
    output_path = Path(args.base_path) / "test_results.txt"
    checkpoint_path = (Path(args.base_path) / args.checkpoint) if args.checkpoint else None
    log_file = Path(args.base_path) / f"test_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    visualise_file = Path(args.base_path)

    # Setup logger
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
    logger.info(f"Testing with path {config_path} and checkpoint {checkpoint_path}")

    # Build model and dataset from config
    factory = Factory(config_path, checkpoint_path)
    model = factory.create_speaker_embedder()
    test_datasets = factory.create_test_datasets()
    device = factory.device
    test_config = factory.config_test

    # Test
    results = evaluate_speaker_verification_spoofed(model=model, datasets=test_datasets, device=device, visualise_save_path=visualise_file, **test_config)

    # Save results to file
    with open(output_path, "w") as f:
        f.write(json.dumps(results, indent=4))
    logger.info(f"Evaluation results saved to {output_path}")

if __name__ == "__main__":
    main()
