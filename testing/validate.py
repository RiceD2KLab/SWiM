"""
This script is used to validate an Ultralytics YOLO model.

Args:
    config (str): The path to the configuration file.
    run_name (str): The name of the run to validate.
"""

from argparse import ArgumentParser
import os
import sys

from ml_collections import ConfigDict
from ultralytics import YOLO

from loguru import logger
from warnings import filterwarnings

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from consts import ROOT_DIR, RUNS_DIR_NAME, CONFIGS_DIR, LOGS_DIR_NAME
from utils.config import get_config

filterwarnings("ignore")


def validate(cfg: ConfigDict, run_name: str):
    """
    Validate an Ultralytics YOLO model.

    Args:
        cfg (ConfigDict): The configuration settings.
        run_name (str): The name of the run to validate.

    Returns:
        None
    """
    # Initialize the Model
    model = YOLO(
        model=ROOT_DIR / RUNS_DIR_NAME / run_name / "weights" / "best.pt",
    )

    # Validate the Model
    metrics = model.val(
        project=RUNS_DIR_NAME,
        name=run_name + "_validation",
    )
    logger.info(f"Validation metrics: {metrics}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    args = parser.parse_args()

    logger.add(
        LOGS_DIR_NAME / f"{args.run_name}_val.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
        rotation="50 MB",
        retention="20 days",
    )

    cfg = get_config(CONFIGS_DIR / args.config_file)
    validate(cfg, args.run_name)
