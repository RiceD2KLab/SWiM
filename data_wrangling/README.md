# Data Wrangling

This directory contains scripts and tools for preprocessing data for the NASA Segmentation F24 project.

## Contents

- `preprocessing` directory: contains scripts to convert data into readily usable YOLO format and to perform other preprocessing steps such as generating masks, resizing images and combining datasets
    - `generate_posebowl_masks.py`: Script for generating segmentation masks for the spacecrafts in Posebowl_ObjDet dataset using Meta's SAM2.
    - `binary_masks_to_yolo_polys.py`: Script for converting binary segmentation masks to yolo-polygon coordinates.
    - `resize_and_merge_classes_spacecrafts.py`: Script for resizing the images and masks to (1280, 1024) using Lanczos interpolation. It also merges the classes in the masks into one.
    - `process_spacecrafts_datatsets.py`: Script is similar to `resize_and_merge_classes_spacecrafts.py` but also checks dimension and corresponding masks

- `utils` directory: contains scripts to create yaml file and util function script
    - `create_yaml.py`: Script for generating the yaml file required to train ultralytics models.
    - `utils.py`: function `yolo_polygon_to_xyxy` which converts YOLO polygon format to YOLO bounding box format. Used to generate plot through function `analyze_bbox_distribution`.
- `plots_&_EDA` directory: contains script to analyze spacecraft dataset and perform segmentation.
    - `plots_posebowl_dataset.py`: Majorly generates plots to analyse the Field-Of-View(FOV) of the spacecraft with repect to the overall size of the image and also svaluates mask quality.



## Usage

1. **Segmentation Masks Generation**: Run `generate_posebowl_masks.py` to generate segmentation masks for Posebowl_ObjDet dataset using SAM2.
    
    This script expects `posebowl_object` dataset as input with the following directory structure. Here, `labels/` contain the bounding box coordinates for each image in `images/` folder in `.txt` format.
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
4. **Process Spacecrafts Dataset**: Run `process_spacecrafts_datatsets.py` to 

    - Check if all images and masks are in PNG format.
    - Verifie that images and masks have the same dimensions.
    - Ensure the number of images and masks match for each dataset split.
    - Check if every image has a corresponding mask.
    - Resize images and masks to 1280x1024 pixels.
    - Merge mask classes into a binary mask.
    - Save processed images and masks in a new directory structure.

    Since it is similar to `resize_and_merge_classes_spacecrafts.py` run only either of them
    ```sh
    python process_spacecrafts_datatsets.py --dir_name spacecrafts
    ```

5. **Create Plots for analysing Posebowl dataset**: Run `plots_posebowl_dataset.py` to generate plots and csv files to analyse data

    This script expects the `data_dir` to have `images/` and `labels/` subdirectories.
            - data_dir (str): Path to the main data directory
            - models_dir (str): Path to the directory containing the SAM model
            - output_dir (str): Path to save plots and CSV files
            - sample_size (int): Number of images to sample for segmentation
            - bbox_threshold (float): Threshold for bounding box area as a fraction of image area
            - is_polygon (bool): Flag indicating if labels are in polygon format
            - img_width (int): Width of the images in pixels
            - img_height (int): Height of the images in pixels
    ```sh
    python plots_posebowl_dataset.py --data_dir posebowl_segmented --models_dir sam_model --output_dir /plots --sample_size 10 --bbox_threshold 0.5 --is_polygon True --img_width 1280 --img_height 1024
    ```


6. **Create YAML**: Run `create_yaml.py` to generate the YAML file necessary for training ultralytics models.

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

This project is licensed under the Apache License. See the [LICENSE](../LICENSE) file for details.

