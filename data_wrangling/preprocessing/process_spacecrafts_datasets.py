"""
Spacecraft Image and Mask Processing Script

This script processes spacecraft images and masks, performing various checks and preprocessing steps.

Usage:
    Run the script from the command line with the data directory as an argument:
    python script_name.py /path/to/data/directory

The script performs the following tasks:
1. Checks if all images and masks are in PNG format.
2. Verifies that images and masks have the same dimensions.
3. Ensures the number of images and masks match for each dataset split.
4. Checks if every image has a corresponding mask.
5. Resizes images and masks to 1280x1024 pixels.
6. Merges mask classes into a binary mask.
7. Saves processed images and masks in a new directory structure."""


import os
import sys
from pathlib import Path
import cv2
from PIL import Image
from tqdm import tqdm
import argparse

def process_images(data_dir):
    """    process_images(data_dir: str) -> None:
                  Main function to process spacecraft images and masks."""
    
    spacecrafts_dir = Path(data_dir) / "spacecrafts"
    images_dir = spacecrafts_dir / "images"
    masks_dir = spacecrafts_dir / "mask"

    train_images = os.listdir(images_dir / "train")
    train_masks = os.listdir(masks_dir / "train")
    val_images = os.listdir(images_dir / "val")
    val_masks = os.listdir(masks_dir / "val")

    # Check if all images and masks are in the png format
    for img in train_images + val_images:
        assert img.endswith(".png"), f"{img} is not a png file"
    print("All images are in png format")
        
    for mask in train_masks + val_masks:
        assert mask.endswith(".png"), f"{mask} is not a png file"
    print("All masks are in png format")

    # Check if all images and masks are of the same size
    check_image_sizes(images_dir, masks_dir, train_images, "train")
    check_image_sizes(images_dir, masks_dir, val_images, "val")

    # Check if the number of images and masks are the same
    assert len(train_images) == len(train_masks), "Number of training images and masks are not the same"
    assert len(val_images) == len(val_masks), "Number of validation images and masks are not the same"
    print("Number of training images and masks:", len(train_images))

    # Check if every image has a corresponding mask
    check_corresponding_masks(train_images, train_masks, "train")
    check_corresponding_masks(val_images, val_masks, "val")

    dest_dir = Path(data_dir) / "spacecrafts_processed"
    images_dest_dir = dest_dir / "images"
    masks_dest_dir = dest_dir / "masks"

    os.makedirs(images_dest_dir / "train", exist_ok=True)
    os.makedirs(masks_dest_dir / "train", exist_ok=True)
    os.makedirs(images_dest_dir / "val", exist_ok=True)
    os.makedirs(masks_dest_dir / "val", exist_ok=True)

    # Copy and resize train images and masks
    process_and_copy_images(images_dir, masks_dir, images_dest_dir, masks_dest_dir, train_images, "train")
    process_and_copy_images(images_dir, masks_dir, images_dest_dir, masks_dest_dir, val_images, "val")

def check_image_sizes(images_dir, masks_dir, image_list, split):
    """check_image_sizes(images_dir: Path, masks_dir: Path, image_list: List[str], split: str) -> None:
            Checks if all images and masks in a given split have the same size."""

    for img in tqdm(image_list, desc=f"Checking {split} image sizes"):
        img_path = images_dir / split / img
        mask_path = masks_dir / split / img.replace(".png", "_mask.png")
        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path))
        
        if img is None:
            print(f"Could not read image: {img_path}")
        if mask is None:
            print(f"Could not read mask: {mask_path}")
        
        img_size = img.shape[:2]
        mask_size = mask.shape[:2]
        assert img_size == mask_size, f"[{split}] Image and mask sizes do not match for {img}"
    print(f"All {split} images and masks are of the same size")

def check_corresponding_masks(images, masks, split):
    """check_corresponding_masks(images: List[str], masks: List[str], split: str) -> None:
            Verifies that each image has a corresponding mask file."""
    for img in images:
        assert f"{os.path.splitext(img)[0]}_mask.png" in masks, f"Mask for {img} not found in {split} set"
    print(f"All {split} images have a corresponding mask")

def process_and_copy_images(images_dir, masks_dir, images_dest_dir, masks_dest_dir, image_list, split):
    """process_and_copy_images(images_dir: Path, masks_dir: Path, images_dest_dir: Path, 
                            masks_dest_dir: Path, image_list: List[str], split: str) -> None:
        Resizes and copies images and masks to a new directory."""

    for img in tqdm(image_list, desc=f"Copying {split} images"):
        img_path = images_dir / split / img
        mask_path = masks_dir / split / img.replace(".png", "_mask.png")
        
        img_dest_path = images_dest_dir / split / img
        mask_dest_path = masks_dest_dir / split / img
            
        # Resize image
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (1280, 1024), interpolation=cv2.INTER_LANCZOS4)
        img = Image.fromarray(img)
        img.save(img_dest_path)
        
        # Merge classes and Resize mask
        mask = cv2.imread(str(mask_path))    
        mask = cv2.resize(mask, (1280, 1024), interpolation=cv2.INTER_LANCZOS4)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask[mask > 0] = 255
        
        mask = Image.fromarray(mask)
        mask.save(mask_dest_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process spacecraft images and masks.")
    parser.add_argument("data_dir", type=str, help="Path to the data directory")
    args = parser.parse_args()

    process_images(args.data_dir)
