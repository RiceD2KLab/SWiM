# Utils

This directory contains scripts and tools for training and utilizing the segmentation models for the NASA Segmentation F24 project.

## Contents

- `config.py`: A utility script that contains helper functions, such as loading configuration files, used throughout the project. The key function, `get_config`, is used in the training script to load the necessary parameters from a YAML file.

## Usage

1. **Utility Functions in `config.py`**: 
   The `config.py` script provides utility functions that simplify the training process. Specifically, the `get_config` function is used in the `train.py` script to load the necessary training configurations from a YAML file. 

   **`get_config(config_file: str) -> ConfigDict`**:
   - **Purpose**: This function loads the YAML configuration file specified in the command line when running `train.py`.
   - **Usage in `train.py`**: The function loads the configuration file, which contains important settings such as model name, dataset path, and training parameters.

   ```python
   from utils import get_config
   config = get_config("config.yaml")
   ```

   The configuration returned by `get_config` is used to set up the YOLO model training.

