# Benchmarking YOLO Model Inference on a Resource-Constrained Environment

## Overview

This repository contains a setup for simulating resource constraints of an **UP Board's Intel® Atom™ x5-Z8350 SoC** on a laptop with a **13th Gen Intel(R) Core(TM) i9-13900H** CPU. The goal is to benchmark the inference speed of an Ultralytics YOLO model for instance segmentation in various formats, including ***ONNX, OpenVINO, Torch, and TorchScript***.

## Directory Structure

```
runtime/
├── submissions/
│   ├── configs/
│   │   └── config.yaml
│   ├── data/
│   │   ├── images/val/
│   │   ├── labels/val/
│   │   └── data.yaml
│   ├── logs/
│   └── models/
├── tests/
│   ├── test_environment.py
│   └── test_packages.py
├── benchmark.py
├── Dockerfile
├── README.md
├── requirements.txt
├── run_benchmark.sh
└── run_container.sh
```

## Getting Started

### Prerequisites

- Docker installed on your system.
- An Ultralytics YOLO `.pt` model file you want to benchmark, placed in the models directory.

### Setup

1. Place your YOLO model in the `submissions/models/` directory.

2. Update the `submissions/configs/config.yaml` file:

   ```yaml
   model_name: "<your_model_name>" # Name of the .pt file
   benchmark:
     device: "cpu" # or [0] for GPU
     img_size: 640
     fp16_quant: False
     int8_quant: False
     verbose: True
   ```

3. Prepare your test data:

   - Place your test images in `submissions/data/images/val/` as `.png` files.
   - Place corresponding labels in `submissions/data/labels/val/` as `.txt` files.

### Building the Docker Image

To build the Docker image, run the following command from the root directory of the project:

```bash
docker build -t yolo_benchmark .
```

### Running the Benchmark

Once the image is built, you can run the benchmark using:

```bash
sh ./run_container.sh
```

### Resource Constraint Simulation

To accurately benchmark the inference speed of the YOLO model under resource-constrained conditions, we simulate the limited CPU capabilities of an **UP Board's Intel® Atom™ x5-Z8350 SoC**. This is achieved by setting a CPU limit when running the Docker container. The command used in the script `run_container.sh` is:

```bash
docker run --cpus="1.4" --memory="4g" -v "$(pwd)/submissions:/code_execution/submissions/" yolo_benchmark
```

The **Intel® Atom™ x5-Z8350 SoC** is designed for low power consumption and is typically used in embedded systems. By constraining the CPU resources to *~1.4 cores* and *4GB RAM*, we aim to create a realistic environment that mimics the performance of the target hardware.

### Testing the Environment

The Docker image will automatically run tests to verify the environment when it starts. Ensure that all necessary packages and configurations are correctly set up.

## Scripts

- **run_benchmark.sh**: Script to execute the benchmark process.
- **run_container.sh**: A convenient script to run the Docker container.

## License

This project is licensed under the APACHE License - see the [LICENSE](../../LICENSE) file for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/) for the YOLO model implementation.
- Docker for containerization.
