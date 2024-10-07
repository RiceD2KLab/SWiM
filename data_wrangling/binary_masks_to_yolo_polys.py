"""
    This script converts binary masks to YOLO polygons. 
    The binary masks are loaded from the masks directory. 
    
    Args:
        source_dir (str): The directory containing the binary masks folder in the data directory.
        dest_dir (str): The directory to save the YOLO labels in the data directory.
        
    Results:
        The YOLO polygons are saved in the labels directory @ dest_dir. 
"""

from argparse import ArgumentParser

import os
import sys
from typing import Tuple
from loguru import logger
import cv2
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from consts import DATA_DIR


def binary_mask_to_yolo_poly(mask_path: str) -> Tuple[str]:
    """
    Convert a binary mask to a YOLO polygon string.

    Args:
        mask_path (str): path to the mask.

    Returns:
        Tuple[str]: List of YOLO polygon strings.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    size = mask.shape[::-1]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        cnt = cnt.squeeze() / size
        cnt = cnt.flatten().tolist()
        polygons.append(" ".join(map(str, cnt)))

    return polygons


def main(source_dir: str, dest_dir: str) -> None:
    masks_dir = DATA_DIR / source_dir / "masks"
    labels_dir = DATA_DIR / dest_dir / "labels"

    logger.info("Converting binary masks to YOLO polygons.")
    logger.debug(f"Source directory: {masks_dir.resolve()}")
    logger.debug(f"Destination directory: {labels_dir.resolve()}")

    for split in ["train", "test", "val"]:
        os.makedirs(labels_dir / split, exist_ok=True)
        data_points = os.listdir(masks_dir / split)

        for mask in tqdm(data_points, desc=split):
            mask_path = masks_dir / split / mask
            label_path = labels_dir / split / mask.replace(".png", ".txt")

            polygons = binary_mask_to_yolo_poly(mask_path)
            with open(label_path, "w", encoding="utf-8") as f:
                for poly in polygons:
                    f.write(f"0 {poly}\n")
        logger.info(f"Finished converting {split} masks to YOLO polygons.")
    logger.info("Finished converting binary masks to YOLO polygons.")


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="The directory containing the binary masks folder in the data directory.",
    )
    argparser.add_argument(
        "--dest_dir",
        type=str,
        required=True,
        help="The directory to save the YOLO labels in the data directory",
    )
    args = argparser.parse_args()

    main(args.source_dir, args.dest_dir)
