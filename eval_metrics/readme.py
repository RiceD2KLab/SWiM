# Segmentation Evaluation Script

## Overview
This script evaluates segmentation performance using a trained YOLO model. It compares predicted segmentation masks against ground truth masks derived from YOLO format TXT files. The evaluation metrics calculated are the Dice coefficient and the Hausdorff distance, which provide insights into the accuracy and similarity of the segmentations.

## Metrics Description

### Dice Coefficient
The Dice coefficient is a statistical measure used to gauge the similarity between two sets. It ranges from 0 to 1, where 1 indicates perfect agreement between the predicted and true masks. The formula for calculating the Dice coefficient is:

Dice = (2 * |A ∩ B|) / (|A| + |B|)


Where:
- |A| is the area of the predicted mask.
- |B| is the area of the true mask.
- |A ∩ B| is the area of overlap between the predicted and true masks.

### Hausdorff Distance
The Hausdorff distance measures how far two subsets of a metric space are from each other. It quantifies the maximum distance of a set to the nearest point in another set. A smaller Hausdorff distance indicates that the predicted mask closely matches the true mask.

## Functions

### create_mask_from_polygons(image_shape, polygons)
Creates a binary mask from a list of polygon coordinates.

**Parameters:**
- `image_shape`: Tuple representing the shape of the image (height, width).
- `polygons`: List of polygons where each polygon is represented by its vertex coordinates.

### process_yolo_txt_file(txt_file_path, image_shape)
Processes a YOLO format TXT file to extract polygon coordinates and create a corresponding binary mask.

**Parameters:**
- `txt_file_path`: Path to the YOLO format TXT file.
- `image_shape`: Tuple representing the shape of the image (height, width).

**Returns:**
- A binary mask image corresponding to the polygons defined in the TXT file.

### get_pred_and_true(image_path, txt_file_path, model, og_shape=(1024, 1280))
Obtains predicted and true masks for a given image.

**Parameters:**
- `image_path`: Path to the input image.
- `txt_file_path`: Path to the corresponding YOLO format TXT file.
- `model`: The trained YOLO model used for prediction.
- `og_shape`: Original shape of images (default: (1024, 1280)).

**Returns:**
- A tuple containing:
  - `y_true`: True mask derived from ground truth.
  - `y_pred`: Predicted mask from the YOLO model.

### dice(pred, true, k=255)
Calculates the Dice coefficient between two binary masks.

**Parameters:**
- `pred`: Predicted mask.
- `true`: True mask.
- `k`: Value representing foreground pixels (default: 255).

**Returns:**
- The Dice coefficient as a float.

### hausdorff_distance_mask(y_true, y_pred)
Calculates Hausdorff distance between two segmentation masks.

**Parameters:**
- `y_true`: True mask.
- `y_pred`: Predicted mask.

**Returns:**
- The Hausdorff distance as a float.

## Command-Line Arguments

To run this script, use the following command:

```bash
python3 dice_and_hausdorff.py --image_dir /path/to/images --txt_dir /path/to/txt_files --model_path /path/to/model.pt --log_dir /path/to/save/logs/
```

**Arguments:**
--image_dir (str): Directory containing input images. Required.
--txt_dir (str): Directory containing corresponding YOLO format TXT files. Required.
--model_path (str): Path to the trained YOLO model. Required.
--log_dir (str): Directory to save log file (default: root directory). Optional.

