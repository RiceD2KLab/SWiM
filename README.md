
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

License
This project is licensed under the MIT License. See the LICENSE file for details.

