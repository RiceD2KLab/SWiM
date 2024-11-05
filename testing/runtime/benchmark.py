"""
This script benchmarks an Ultralytics YOLO model using the Ultralytics benchmarking function.

Usage:
    python benchmark.py
"""

import time
from pathlib import Path
import yaml

from ml_collections import ConfigDict

from ultralytics import YOLO
from ultralytics.cfg import TASK2DATA, TASK2METRIC
from ultralytics.utils import (
    ASSETS,
    LOGGER,
    WEIGHTS_DIR,
)
from ultralytics.utils.checks import check_yolo
from ultralytics.utils.files import file_size
from ultralytics.utils.torch_utils import select_device

from loguru import logger


def export_formats():
    """Ultralytics YOLO export formats."""
    x = [
        ["ONNX", "onnx", ".onnx", True, True],
        ["OpenVINO", "openvino", "_openvino_model", True, False],
        ["PyTorch", "-", ".pt", True, True],
        ["TorchScript", "torchscript", ".torchscript", True, True],
    ]
    return dict(zip(["Format", "Argument", "Suffix", "CPU", "GPU"], zip(*x)))


def benchmark(
    model=WEIGHTS_DIR / "yolo11n.pt",
    data=None,
    imgsz=160,
    half=False,
    int8=False,
    device="cpu",
    verbose=False,
    eps=1e-3,
):
    """
    Benchmark a YOLO model across different formats for speed and accuracy.

    Args:
        model (str | Path): Path to the model file or directory.
        data (str | None): Dataset to evaluate on, inherited from TASK2DATA if not passed.
        imgsz (int): Image size for the benchmark.
        half (bool): Use half-precision for the model if True.
        int8 (bool): Use int8-precision for the model if True.
        device (str): Device to run the benchmark on, either 'cpu' or 'cuda'.
        verbose (bool | float): If True or a float, assert benchmarks pass with given metric.
        eps (float): Epsilon value for divide by zero prevention.

    Returns:
        (pandas.DataFrame): A pandas DataFrame with benchmark results for each format, including file size, metric,
            and inference time.

    Examples:
        Benchmark a YOLO model with default settings:
        >>> from ultralytics.utils.benchmarks import benchmark
        >>> benchmark(model="yolo11n.pt", imgsz=640)
    """
    import pandas as pd  # scope for faster 'import ultralytics'

    pd.options.display.max_columns = 10
    pd.options.display.width = 120
    device = select_device(device, verbose=False)
    if isinstance(model, (str, Path)):
        model = YOLO(model)

    y = []
    t0 = time.time()
    for i, (name, format, suffix, cpu, gpu) in enumerate(
        zip(*export_formats().values())
    ):
        emoji, filename = "❌", None  # export defaults
        try:
            # Checks
            if "cpu" in device.type:
                assert cpu, "inference not supported on CPU"
            if "cuda" in device.type:
                assert gpu, "inference not supported on GPU"

            # Export
            if format == "-":
                filename = model.ckpt_path or model.cfg
                exported_model = model  # PyTorch format
            else:
                filename = model.export(
                    imgsz=imgsz,
                    format=format,
                    half=half,
                    int8=int8,
                    device=device,
                    verbose=False,
                )
                exported_model = YOLO(filename, task=model.task)
                assert suffix in str(filename), "export failed"
            emoji = "❎"  # indicates export succeeded

            # Predict
            exported_model.predict(
                ASSETS / "bus.jpg", imgsz=imgsz, device=device, half=half
            )

            # Validate
            data = (
                data or TASK2DATA[model.task]
            )  # task to dataset, i.e. coco8.yaml for task=detect
            key = TASK2METRIC[
                model.task
            ]  # task to metric, i.e. metrics/mAP50-95(B) for task=detect
            results = exported_model.val(
                data=data,
                batch=1,
                imgsz=imgsz,
                plots=False,
                device=device,
                half=half,
                int8=int8,
                verbose=False,
            )
            metric, speed = results.results_dict[key], results.speed["inference"]
            fps = round(1000 / (speed + eps), 2)  # frames per second
            y.append(
                [
                    name,
                    "✅",
                    round(file_size(filename), 1),
                    round(metric, 4),
                    round(speed, 2),
                    fps,
                ]
            )
        except Exception as e:
            if verbose:
                assert type(e) is AssertionError, f"Benchmark failure for {name}: {e}"
            LOGGER.warning(f"ERROR ❌️ Benchmark failure for {name}: {e}")
            y.append(
                [name, emoji, round(file_size(filename), 1), None, None, None]
            )  # mAP, t_inference

    # Print results
    check_yolo(device=device)  # print system info
    df = pd.DataFrame(
        y,
        columns=[
            "Format",
            "Status❔",
            "Size (MB)",
            key,
            "Inference time (ms/im)",
            "FPS",
        ],
    )

    name = Path(model.ckpt_path).name
    s = f"\nBenchmarks complete for {name} on {data} at imgsz={imgsz} ({time.time() - t0:.2f}s)\n{df}\n"
    with open("benchmarks.log", "a", errors="ignore", encoding="utf-8") as f:
        f.write(s)

    if verbose and isinstance(verbose, float):
        metrics = df[key].array  # values to compare to floor
        floor = verbose  # minimum metric floor to pass, i.e. = 0.29 mAP for YOLOv5n
        assert all(
            x > floor for x in metrics if pd.notna(x)
        ), f"Benchmark failure: metric(s) < floor {floor}"

    return df


def benchmark_model(config: ConfigDict) -> None:
    """
    Benchmark an Ultralytics YOLO model.

    Args:
        config (ConfigDict): The configuration settings.

    Returns:
        None
    """
    logger.info(f"Benchmarking the YOLO model: {config.model_name}")
    logger.info(f"Configuration: {config.benchmark}")

    results = benchmark(
        model=f"/code_execution/submissions/models/{config.model_name}.pt",
        data="/code_execution/submissions/data/data.yaml",
        device=config.benchmark.device,
        imgsz=config.benchmark.img_size,
        half=config.benchmark.fp16_quant,
        int8=config.benchmark.int8_quant,
        verbose=config.benchmark.verbose,
    )

    logger.info(f"Benchmark Results: \n{results}")
    results.to_csv(
        f"/code_execution/submissions/logs/{config.model_name}_benchmark_results.csv"
    )


def get_config() -> ConfigDict:
    """
    Get the configuration settings from the config.yaml file.

    Returns:
        ConfigDict: The configuration settings.
    """
    config_path = "/code_execution/submissions/configs/config.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        config = ConfigDict(yaml.safe_load(file))

    return config


if __name__ == "__main__":
    cfg = get_config()

    logger.add(
        f"/code_execution/submissions/logs/{cfg.model_name}_benchmark.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
        rotation="50 MB",
        retention="20 days",
    )

    benchmark_model(cfg)
