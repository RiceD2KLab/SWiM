# Overview
=====================================

The purpose of the 'create_yaml.py' file here is to create a YOLOv8 training YAML file in a specified file path.

## Usage
-------------

### Arguments

The `create_yaml.py` script takes three mandatory arguments:

1. **Path to Save the YAML File**:
   - The directory path where the generated YAML file will be saved.
2. **Path to the Dataset Directory**:
   - The directory path containing the dataset used for training.
3. **Data Augmentation Flag**:
   - A boolean flag indicating whether data augmentation should be enabled (`True`) or disabled (`False`).

### Example Command

To run the script, use the following command format:

```bash
python create_yaml.py /path/to/save/yaml /path/to/dataset True
