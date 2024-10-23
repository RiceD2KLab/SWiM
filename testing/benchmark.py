from argparse import ArgumentParser
import os
import sys

from ml_collections import ConfigDict
from ultralytics.utils.benchmarks import benchmark

from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from consts import ROOT_DIR, RUNS_DIR_NAME, CONFIGS_DIR, DATA_DIR, LOGS_DIR_NAME
from utils.config import get_config


def benchmark_model(cfg, run_name):
    """
    Benchmark an Ultralytics YOLO model.

    Args:
        cfg (ConfigDict): The configuration settings.
        run_name (str): The name of the run to benchmark.

    Returns:
        None
    """
    logger.info(f"Benchmarking the YOLO model: {run_name}")
    logger.info(f"Configuration: \n{cfg.benchmark}")
    # Benchmark the Model
    benchmark(
        model=ROOT_DIR / RUNS_DIR_NAME / run_name / "weights" / "best.pt",
        data=DATA_DIR / cfg.dataset.path / "data.yaml",
        imgsz=cfg.dataset.img_size,
        device=cfg.benchmark.device,
        half=cfg.benchmark.fp16_quant,
        int8=cfg.benchmark.int8_quant,
        verbose=cfg.benchmark.verbose,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    args = parser.parse_args()

    cfg = get_config(CONFIGS_DIR / args.config_file)

    logger.add(
        ROOT_DIR / LOGS_DIR_NAME / f"{args.run_name}_benchmark.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
        rotation="50 MB",
        retention="20 days",
    )

    benchmark_model(cfg, args.run_name)
