# Overview
=====================================
This script is designed to profile the execution of a YOLO (You Only Look Once) model using PyTorch and the Ultralytics YOLO library. It allows users to measure the performance of the model on various input sizes and configurations, including the option to use CUDA for accelerated profiling.

## Usage
-------------

To use this script, you need to have the following dependencies installed:

- **Python**: Ensure you have Python installed on your system.
- **PyTorch**: Install PyTorch using `pip install torch`.
- **Ultralytics YOLO**: Install the Ultralytics YOLO library using `pip install ultralytics`.

### Command-Line Arguments

The script takes the following command-line arguments:

#### `--yolo-path`
- **Description**: The path to the YOLO model file.
- **Required**: Yes

#### `--batch-size`
- **Description**: The batch size of the input tensor.
- **Default**: 1
- **Type**: Integer

#### `--height`
- **Description**: The height of the input tensor.
- **Default**: 1024
- **Type**: Integer

#### `--width`
- **Description**: The width of the input tensor.
- **Default**: 1280
- **Type**: Integer

#### `--cuda`
- **Description**: Use CUDA for profiling if available.
- **Default**: False (CPU)
- **Type**: Boolean (action='store_true')

### Example Command

To run the script, use the following command format:

```bash
python profile_yolo.py --yolo-path /path/to/yolo/model --batch-size 4 --height 640 --width 640 --cuda
