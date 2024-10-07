# Data Wrangling

This directory contains scripts and tools for preprocessing data for the NASA Segmentation F24 project.

## Contents

- `generate_posebowl_masks.py`: Script for generating segmentation masks for the spacecrafts in Posebowl_ObjDet dataset using Meta's SAM2.
- `binary_masks_to_yolo_polys.py`: Script for converting binary segmentation masks to yolo-polygon coordinates.
- `resize_and_merge_classes_spacecrafts.py`: Script for resizing the images and masks to (1280, 1024) using Lanczos interpolation. It also merges the classes in the masks into one.
- `create_yaml.py`: Script for generating the yaml file required to train ultralytics models.

## Usage

1. **Segmentation Masks Generation**: Run `generate_posebowl_masks.py` to generate segmentation masks for Posebowl_ObjDet dataset using SAM2.
    
    This script expects `posebowl_objdet` dataset as input with the following directory structure. Here, `labels/` contain the bounding box coordinates for each image in `images/` folder in `.txt` format.
    ```
    /data/posebowl_objdet/
    ├── images/
    |   ├── train
    |   ├── test
    |   └── val 
    ├── labels/
    |   ├── train
    |   ├── test
    |   └── val
    ```

    ```sh
    python generate_posebowl_masks.py --source_dir posebowl_objdet --dest_dir posebowl_segmented --split train --sam_size l
    ```

2. **YOLO Polygon Coordinates Generation**: Run `binary_masks_to_yolo_polys.py` to convert the binary masks into YOLO polygon coordinates.

    This script expects the binary masks to be present in `/source_dir/masks/[train/test/val]` folders. The polygon coordinates are then saved under `/dest_dir/labels/[train/test/val]` folders.
    ```sh
    python binary_masks_to_yolo_polys.py --source_dir posebowl_segmented --dest_dir posebowl_segmented
    ```

3. **Resize Spacecrafts images + masks and Merge classes**: Run `resize_and_merge_classes_spacecrafts.py` to resize the images and masks to the same size as posebowl dataset. It also merges the classes of the masks into one.
    
    This script expects the spacecrafts data dir name in the `/data/` dir.
    ```sh
    python resize_and_merge_classes_spacecrafts.py --dir_name spacecrafts
    ```
4. **Create YAML**: Run `create_yaml.py` to generate the YAML file necessary for training ultralytics models.

    This script expects the `data_dir` to have `images/` and `labels/` subdirectories.
    ```sh
    python create_yaml.py --data_dir posebowl_segmented
    ```

## Requirements

[![Python 3.11+](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)


Install the required packages from the project's root directory using:
```sh
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.

