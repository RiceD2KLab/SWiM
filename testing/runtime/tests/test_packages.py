"""
This script tests importing the required packages for the runtime environment.

Usage:
    python test_packages.py
"""

import importlib

packages = [
    "numpy",
    "cv2",  # opencv-python
    "PIL",  # Pillow
    "matplotlib",
    "torch",
    "torchvision",
    "ml_collections",
    "ultralytics",
    "transformers",
    "loguru",
    "onnx",
    "onnxruntime",
]


def test_imports():
    """
    Test importing the required packages.
    """
    for package in packages:
        try:
            importlib.import_module(package)
            print(f"Successfully imported {package}")
        except ImportError as e:
            print(f"Error importing {package}: {e}")
            raise


if __name__ == "__main__":
    test_imports()
