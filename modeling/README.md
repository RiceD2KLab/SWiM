# Modeling 

This directory includes essential scripts for setting up and managing the training process for the NASA Segmentation F24 project.

## Contents

- `train.py`: Script for training the YOLO segmentation model using the Ultralytics library. This script loads configurations, initializes the model, and performs the training process, saving the results in the `runs/` directory.


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

