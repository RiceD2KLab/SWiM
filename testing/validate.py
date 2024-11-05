"""
This script is used to validate an Ultralytics YOLO model on either val or test set.

Args:
    config (str): The path to the configuration file.
    model_file (str): The name of the model file to validate under models/ dir.
    split (str): The split to validate on (val or test).
    log_to_wandb (bool): Whether to log the validation results to Weights & Biases.
"""

from argparse import ArgumentParser
import os
import sys
from warnings import filterwarnings

from ml_collections import ConfigDict
from ultralytics import YOLO

from dotenv import load_dotenv
from loguru import logger
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from consts import ROOT_DIR, RUNS_DIR_NAME, MODELS_DIR, CONFIGS_DIR, LOGS_DIR_NAME
from utils.config import get_config

filterwarnings("ignore")


def validate(
    cfg: ConfigDict, model_file: str, split: str, log_to_wandb: bool = False
) -> None:
    """
    Validate an Ultralytics YOLO model.

    Args:
        cfg (ConfigDict): The configuration settings.
        model_file (str): The name of the model file to validate under models/ dir.
        split (str): The split to validate on (val or test).
        log_to_wandb (bool): Whether to log the validation results to Weights & Biases.

    Returns:
        None
    """
    logger.info(f"Validating model: {model_file}")
    if log_to_wandb:
        load_dotenv()
        wandb.login(os.getenv("WANDB_API_KEY"))
        wandb.init(
            project="NASA_SEG",
            name=f"{cfg.experiment.name}_{split}",
            job_type="validation",
        )

    # Initialize the Model
    model = YOLO(
        model=MODELS_DIR / model_file,
    )

    # Validate the Model
    metrics = model.val(
        split=split,
        project=RUNS_DIR_NAME,
        name=f"{cfg.experiment.name}_{split}",
    )
    logger.info(f"Validation metrics: \n{metrics.results_dict}")

    if log_to_wandb:
        experiments = os.listdir(ROOT_DIR / RUNS_DIR_NAME)
        val_experiments = [
            dir for dir in experiments if f"{cfg.experiment.name}_{split}" in dir
        ]
        current_val_exp = sorted(val_experiments)[-1]
        val_dir = ROOT_DIR / RUNS_DIR_NAME / current_val_exp

        # Log the Validation Images
        logger.info(f"Logging validation images from {val_dir}")
        imgs = os.listdir(val_dir)
        media = {}

        for img in imgs:
            media[os.path.splitext(img)[0]] = wandb.Image(str(val_dir / img))

        wandb.log(media)

        # Log the Validation Metrics
        wandb.log(metrics.results_dict)
        # Log the Validation Confusion Matrix
        wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--log_to_wandb", action="store_true", default=False)
    args = parser.parse_args()

    cfg = get_config(CONFIGS_DIR / args.config_file)
    logger.add(
        ROOT_DIR / LOGS_DIR_NAME / f"{cfg.experiment.name}_{args.split}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
        rotation="50 MB",
        retention="20 days",
    )
    logger.info(f"Loaded configuration: {args.config_file}")
    logger.info(f"Model file: {args.model_file}")
    logger.info(f"Logging to Weights & Biases: {args.log_to_wandb}")
    logger.info(f"Validation split: {args.split}")

    validate(
        cfg,
        args.model_file,
        args.split,
        args.log_to_wandb,
    )
