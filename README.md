
# Segmentation of an unknown spacecraft for In-space Inspection

## Project Description
This project aims to develop a real-time segmentation algorithm for in-space inspection of spacecraft using deep learning techniques. The focus is to build a general-purpose instance segmentation model that can accurately detect and mask spacecraft components. The model will be trained on synthetic and real datasets using the YOLOv8 nano model, optimized to run on resource-constrained hardware. This algorithm is expected to enhance NASA’s capability for autonomous inspections, improving spacecraft navigation, pose estimation, and structural analysis under various visual distortions in space imagery.


## Software dependencies
[![Python 3.11+](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
- Ultralytics YOLOv8
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Segment Anything Model (SAM2)
- TQDM (for progress bars)
- Loguru (for logging)

## Setup

1. Create a Python `venv` environment

    ```sh
    python3 -m venv .venv
    ```

2. Activate the environment
   -On Windows:
   
        ```sh
        .\.venv\Scripts\activate
        ```
    -On macOS/Linux:
   
        ```sh
        source .venv/bin/activate
        ```

4. Install requirements

    ```sh
    pip install -r requirements.txt
    ```
## Directory Structure

/segmentation_project/
├── data_wrangling/
│   ├── generate_posebowl_masks.py
│   ├── binary_masks_to_yolo_polys.py
│   ├── resize_and_merge_classes_spacecrafts.py
│   ├── create_yaml.py
│
├── modeling/
│   ├── train.py
│   ├── utils.py
│
├── data/
│   ├── posebowl_objdet/ (dataset directory)
│   └── spacecrafts/ (dataset directory)
│
├── config.yaml (example config file for training)
├── requirements.txt
├── LICENSE
└── README.md (this file)

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.


## Usage
To use onnx_pipeline.py, run the following command:

```sh
python script_name.py --model best.onnx --input input_image.png --output output_segmented_image.jpg

```

