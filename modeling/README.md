# Modeling

This directory contains scripts and tools for training and utilizing the segmentation models for the NASA Segmentation F24 project.

## Contents

- `train.py`: Script for training the YOLO segmentation model using the Ultralytics library. This script loads configurations, initializes the model, and performs the training process, saving the results in the `runs/` directory.
- `utils.py`: A utility script that contains helper functions, such as loading configuration files, used throughout the project. The key function, `get_config`, is used in the training script to load the necessary parameters from a YAML file.

## Usage

1. **Training the YOLO Model**: 
   Run `train.py` to train a segmentation model using the Ultralytics YOLO framework.

   This script expects a YAML configuration file to be passed via the command line, which defines various parameters like the model type, dataset path, number of epochs, and more.

   ```sh
   python train.py --config_file config.yaml
   ```

   The `config_file` should contain:
   - `model.name`: Name of the YOLO model to use.
   - `dataset.path`: Path to the dataset directory that contains the `data.yaml`.
   - Other training parameters like batch size, epochs, etc.

   Example usage:
   ```sh
   python train.py --config_file yolo_config.yaml
   ```

2. **Utility Functions in `utils.py`**: 
   The `utils.py` script provides utility functions that simplify the training process. Specifically, the `get_config` function is used in the `train.py` script to load the necessary training configurations from a YAML file. 

   **`get_config(config_file: str) -> ConfigDict`**:
   - **Purpose**: This function loads the YAML configuration file specified in the command line when running `train.py`.
   - **Usage in `train.py`**: The function is called to load the configuration file, which contains important settings such as model name, dataset path, and training parameters.

   ```python
   from utils import get_config
   config = get_config("config.yaml")
   ```

   The configuration returned by `get_config` is used to set up the YOLO model training.

## Directory Structure

```
/modeling/
├── train.py          # Script to train the YOLO segmentation model
```
