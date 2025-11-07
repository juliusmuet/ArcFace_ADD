# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
from datetime import datetime
from pathlib import Path
import logging
from utils.factory import Factory
from train.evaluation import evaluate_speaker_verification
from utils.utils import setup_logging


def main():
    # Get path to config directory
    parser = argparse.ArgumentParser(description="Evaluate speaker verification model.")
    parser.add_argument("base_path", type=str, help="Base path to the config directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint file name")
    parser.add_argument("--test", type=str, default='vox1_o_clean', help="Model checkpoint file name")
    args = parser.parse_args()

    # Get paths
    config_path = Path(args.base_path) / "config.yml"
    checkpoint_path = (Path(args.base_path) / args.checkpoint) if args.checkpoint else None
    log_file = Path(args.base_path) / f"test_sv_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

    # Setup logger
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
    logger.info(f"Testing SV with trials {args.test} with path {config_path} and checkpoint {checkpoint_path}")

    # Build model and dataset from config
    factory = Factory(config_path, checkpoint_path)
    model = factory.create_speaker_embedder()
    wav_loader = factory.create_preprocessor_evaluation()
    test_paths = factory.base_config.get('speaker_verification_evaluation').get(args.test)
    for k, v in test_paths.items():
        test_paths[k] = Path(v) if isinstance(v, str) else v

    # Test SV
    evaluate_speaker_verification(model, wav_loader, **test_paths)


if __name__ == "__main__":
    main()