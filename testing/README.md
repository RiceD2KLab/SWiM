# Testing YOLO Model Validation

## Overview

This folder contains the script used to validate an Ultralytics YOLO model on either the validation or test dataset. The validation process evaluates the model's performance and can log results to Weights & Biases for further analysis.

## Contents

The `validate.py` script accepts several arguments to facilitate the validation of a YOLO model:

- `--config_file`: The path to the configuration file.
- `--model_file`: The name of the model file to validate under the `models/` directory.
- `--split`: The split to validate on (either `val` or `test`).
- `--log_to_wandb`: A boolean flag indicating whether to log the validation results to Weights & Biases.

The `evaluate_model.py` script evaluates the model given the `model_path`, `metrics` to track, and input `image_dir`

The `YOLOv8Seg.py` defines the Yolov8 class with helper methods to run evaluation.

The `error-analysis/error_analyser.py` has script to infer polygon coordinates for the spacecraft and calculate the error from the ground truth mask.

The `runtime/` subdirectory contains the core components for benchmarking YOLO model inference, providing a structured environment for model's inference speed testing.

## Usage

To validate a YOLO model, run the following command:

```bash
python validate.py --config_file <config_file_name> --model_file <model_file.pt> --split <val_or_test> --log_to_wandb
```

### Example

```
python validate.py --config_file config.yaml --model_file baseline_yolov8n.pt --split val --log_to_wandb
```

## Logs
Validation logs are generated in the `logs/` directory under the project's root, with separate log files for each validation run, helping track the validation metrics and outcomes.


## License

This project is licensed under the Apache License. See the [LICENSE](../LICENSE) file for details.


