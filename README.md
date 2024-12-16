
# Segmentation of an unknown spacecraft for In-space Inspection

## Project Description
This project aims to develop a real-time segmentation algorithm for in-space inspection of spacecraft using deep learning techniques. The focus is to build a general-purpose instance segmentation model that can accurately detect and mask spacecraft components. The model will be trained on synthetic and real datasets using the You Only Look Once (YOLO)v8 nano model, optimized to run on resource-constrained hardware. This algorithm is expected to enhance NASA’s capability for autonomous inspections, improving spacecraft navigation, pose estimation, and structural analysis under various visual distortions in space imagery.


## Software dependencies
[![Python 3.11+](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/) ![CUDA v12.1](https://img.shields.io/badge/CUDA-v12_1-violet)

### Training Dependencies
- numpy==1.23.5
- opencv-python==4.10.0.84
- pillow==10.4.0
- matplotlib==3.7.2
- torch==2.4.1
- torchvision==0.19.1
- ml-collections==0.1.1
- pybboxes==0.1.6
- ultralytics==8.0.238
- transformers==4.45.1
- loguru==0.4.6
- wandb==0.18.5
- python-dotenv==1.0.1

### Inference Dependencies
- numpy==1.23.5
- opencv-python==4.10.0.84
- pillow==10.4.0
- torch==2.4.1
- torchvision==0.19.1
- ultralytics==8.0.238
- onnx==1.17.0
- onnxruntime==1.19.2

## Setup

1. Create a Python `venv` environment

    ```sh
    python -m venv .venv
    ```

2. Activate the environment
   - On Windows:
   
        ```sh
        .\.venv\Scripts\activate
        ```
   - On macOS/Linux:
   
        ```sh
        source .venv/bin/activate
        ```

3. Install requirements

    ```sh
    pip install -r requirements.txt
    ```
    
## Usage
To use onnx_pipeline.py, run the following command:

```sh
python src/onnx_pipeline.py --model best.onnx --input input_image.png --output output_segmented_image.jpg --num_threads 3 --num_streams 1
```
## Hardware for Training

Baseline model was trained on the following configuration:

| **Model**          | YOLOv8 - nano                                      |
|--------------------|----------------------------------------------------|
| **GPU**            | NVIDIA GeForce RTX 4060 Laptop GPU                 |
| **GPU Memory**     | 8GB                                                |
| **CPU**            | 13th Gen Intel(R) Core(TM) i9-13900H (2.60 GHz)    |                                  |
| **RAM**            | 16.0 GB                                            |

Other versions of the model were done on the cloud GPU provider: [modal.com](https://modal.com/)

## Model files and Datasets
All the trained models can be found [here](https://drive.google.com/drive/folders/1WxbuNpZJu50HF27rXzarGAWMtwsZDdTc?usp=drive_link). Download the appropriate pt file and save it under `models/` directory.

All the dataset versions can be found [here](https://drive.google.com/drive/folders/1-vd2KWqrl9Z3fpG7iS_r1L7bfjOlIzFt?usp=drive_link). Download the appropriate dataset, unzip it and save it under the `data/` folder.

Information about each dataset version is given in the README attached with the dataset detailing the makeup of the data and the splits.

## Directory Structure
 ```sh
/NASA_segmentation_F24/
├── data_wrangling/
│   ├── generate_posebowl_masks.py
│   ├── binary_masks_to_yolo_polys.py
│   ├── resize_and_merge_classes_spacecrafts.py
│   └── create_yaml.py
├── modeling/
│   └── train.py
├── testing/
│   ├── benchmark.py
│   └── validate.py
├── utils/
│   └── config.py
├── data/
│   ├── dataset-v1/ 
│   └── dataset-v2/ 
├── configs/
│   └── config.yaml
├── requirements.txt
├── LICENSE
├── CONTRIBUTING
├── .env.example
└── README.md 
 ```


## Sample Output of the YoloV8 Model
After running the YOLOv8 segmentation model, you can expect to receive segmentation masks along with various logs and performance metrics that demonstrate the model's efficiency in detecting and masking spacecraft components in real-time.

**Example Input**

<img width="545" alt="Screen Shot 2024-10-23 at 4 52 30 PM" src="https://github.com/user-attachments/assets/87561e48-4e22-41d9-bd77-8354979e6cfd">

**Example Output**

<img width="551" alt="Screen Shot 2024-10-23 at 4 52 35 PM" src="https://github.com/user-attachments/assets/405feb25-03f9-4478-9739-c6f07ed445f1">

- Segmentation Masks: These masks outline the spacecraft components identified in the images.
- Bounding Boxes: Predicted bounding boxes for detected spacecraft objects.


## Weights & Biases Logging

To enable logging of validation results to Weights & Biases (WandB), you'll need to set your WandB API key in a `.env` file. Follow these steps to configure it:

1. **Create a `.env` file** in the root directory of your project based on the provided example:

   ```sh
   cp .env.example .env
   ```

2. **Open the `.env` file** and add your WandB API key:

   ```env
   WAND_API_KEY = "YOUR_API_KEY"
   ```

   Replace `YOUR_API_KEY` with your actual WandB API key, which you can obtain from your [Weights & Biases account](https://wandb.ai/).

## License

This project is licensed under the APACHE License. See the [LICENSE](../LICENSE) file for details.




