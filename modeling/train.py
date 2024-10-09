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

from ml_collections import ConfigDict
from ultralytics import YOLO

from utils import get_config
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from consts import RUNS_DIR_NAME, MODELS_DIR, DATA_DIR


def train(cfg: ConfigDict) -> None:
    """
    Train an Ultralytics YOLO model.

    Args:
        cfg (ConfigDict): The configuration file.

    Returns:
        None
    """
    logger.info(f"Training the YOLO model: {cfg.model.name}")
    model = YOLO(
        model=MODELS_DIR / cfg.model.name,
    )
    model.train(
        data=DATA_DIR / cfg.dataset.dir_name / "data.yaml",
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
        name=cfg.training.name,
    )


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="The name of the configuration file.",
    )

    args = arg_parser.parse_args()
    config = get_config(args.config_file)
    train(config)
