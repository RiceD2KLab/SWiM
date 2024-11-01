# Instance Segmentation Evaluation Script

This script is designed to evaluate the performance of an instance segmentation model using metrics such as Dice coefficient and Hausdorff distance.

## Requirements

- `argparse`
- `cv2` (OpenCV)
- `numpy`
- `scikit-image`
- 'torch'
- 'ultralytics'

## Usage

To use this script, you need to provide the following arguments:

### Arguments

- **`--image_dir`**: Path to the directory containing the image files.
- **`--txt_dir`**: Path to the directory containing the corresponding TXT files for instance segmentation.
- **`--model_path`**: Path to the YOLO model used for making predictions.

### Example Command

```bash
python script_name.py --image_dir /path/to/images --txt_dir /path/to/txt_files --model_path /path/to/model
