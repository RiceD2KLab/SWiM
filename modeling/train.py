"""
    This module is used to train the Segmentation model.
    
    Args:
        data_dir (str): The directory containing the data.
        model_name (str): The name of the model to train.
        training_args_file (str): The path to the training arguments file.
        
    Results:
        The trained model is saved in the runs/ directory along with the training results and plots.
"""

from argparse import ArgumentParser
import os
import sys

import torch
from ml_collections import ConfigDict
from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from loguru import logger
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from consts import RUNS_DIR_NAME, MODELS_DIR, DATA_DIR, CONFIGS_DIR
from utils.config import get_config


def train(cfg: ConfigDict) -> None:
    """
    Train an Ultralytics YOLO model.

    Args:
        cfg (ConfigDict): The configuration file.

    Returns:
        None
    """

    # Initialize WandB
    wandb.init(project="NASA_SEG", name=cfg.experiment.name, job_type="training")

    # Initialize the Model
    logger.info(f"Training the YOLO model: {cfg.experiment.model}")
    model = YOLO(
        model=MODELS_DIR / cfg.experiment.model,
    )

    # Initialize the WandB Callback
    add_wandb_callback(model, enable_model_checkpointing=True)

    # Train the Model
    model.train(
        data=DATA_DIR / cfg.dataset.path / "data.yaml",
        epochs=cfg.training.epochs,
        batch=cfg.training.batch_size,
        device=cfg.training.device,
        fraction=cfg.training.fraction,
        freeze=cfg.training.freeze,
        amp=cfg.training.amp,
        imgsz=cfg.dataset.img_size,
        plots=cfg.training.plots,
        resume=cfg.training.resume,
        project=RUNS_DIR_NAME,
        name=cfg.experiment.name,
    )

    # Finish the WandB Run
    wandb.finish()


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="The name of the configuration file.",
    )

    logger.debug(f"Available GPU: {torch.cuda.is_available()}")
    logger.debug(f"Current Device: {torch.cuda.current_device()}")

    load_dotenv()

    # Initialize your Weights & Biases environment
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    args = arg_parser.parse_args()
    config = get_config(CONFIGS_DIR / args.config_file)
    train(config)
