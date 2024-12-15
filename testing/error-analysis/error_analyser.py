import cv2
import numpy as np
import onnxruntime as ort
import time
import argparse
import os

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from consts import ROOT_DIR

# adding Folder_2 to the system path
sys.path.insert(0, ROOT_DIR / "src")

from testing.YOLOv8Seg import YOLOv8Seg


class ErrorAnalyser:
    def __init__(self, model, test_data_path):
        self.test_data = test_data_path
        self.model = YOLOv8Seg(model)


if __name__ == "__main__":
    # Argument parsing for command-line options
    parser = argparse.ArgumentParser(description="YOLOv8 Segmentation ONNX Inference")
    parser.add_argument(
        "--model", type=str, default="best.onnx", help="Path to the ONNX model file"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input image directory"
    )
    # parser.add_argument('--output', type=str, default='output_segmented_image.jpg', help='Path to save the output image')
    args = parser.parse_args()

    start_time = time.time()

    # Initialize the model
    model = YOLOv8Seg(args.model)

    # Load input images
    imgs_list = []
    results = []
    if os.path.isdir(args.input):
        with os.scandir(args.input) as imgs_dir:
            for file in imgs_dir:
                if file.is_file() and file.name.endswith(".png"):
                    # imgs_list.append(file.path)
                    img = cv2.imread(file.path)
                    result = model(img, conf_threshold=0.4, iou_threshold=0.45)
    else:
        # imgs_list.append(args.input)
        result = model(args.input, conf_threshold=0.4, iou_threshold=0.45)

    results.append(result)
    # Run inference
    # results = model(imgs_list, conf_threshold=0.4, iou_threshold=0.45)

    # Draw and visualize the result
    for result in results:
        boxes = result.boxes
        segments = result.segments
        speed = result.speed
        print(f"Speed: {speed:.4f}")

    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.2f} seconds")
    # print(f"Output image saved to: {args.output}")
