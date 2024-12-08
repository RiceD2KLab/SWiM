# ONNX & ONNX Runtime

This repository contains the source code for ONNX, the Open Neural Network Exchange. 

## ONNX
The Open Neural Network Exchange (ONNX) is an open format used to represent deep learning models. ONNX is supported by a community of partners who have implemented it in many frameworks and tools.

## ONNX Runtime
ONNX Runtime is a performance-focused scoring engine for Open Neural Network Exchange (ONNX) models. ONNX Runtime has proved to considerably increase performance over multiple models as explained in this blog. It is specifically designed to be light-weight and efficient, and it has proven to be performant across a variety of use-cases.

## Usage
To use onnx_pipeline.py, run the following command:

```sh
python src/onnx_pipeline.py --model best.onnx --input input_image.png --output output_segmented_image.jpg --num_threads 3 --num_streams 1
```