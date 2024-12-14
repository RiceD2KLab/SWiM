"""
    Resize and merge classes of spacecrafts dataset.
    For resizing, Lanczos interpolation over an 8x8 neighborhood is used. 
    This method provides the highest quality but is the slowest.
    
    Args:
        dir_name (str): The name of the spacecrafts directory.
        
    Results:
        Creates a new directory with resized images and masks.
        Images + Masks are resized to (1280, 1024)
        The classes in the masks are merged into one.
"""

from argparse import ArgumentParser
import os
import sys
import cv2
from PIL import Image
from loguru import logger
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from consts import DATA_DIR


def main(dir_name: str) -> None:
    spacecrafts_dir = DATA_DIR / dir_name
    images_dir = spacecrafts_dir / "images"
    masks_dir = spacecrafts_dir / "mask"

    train_images = os.listdir(images_dir / "train")
    val_images = os.listdir(images_dir / "val")
    logger.debug(f"Train images: {len(train_images)}")
    logger.debug(f"Val images: {len(val_images)}")

    dest_dir = DATA_DIR / f"{dir_name}_processed"
    images_dest_dir = dest_dir / "images"
    masks_dest_dir = dest_dir / "masks"

    os.makedirs(images_dest_dir / "train", exist_ok=True)
    os.makedirs(masks_dest_dir / "train", exist_ok=True)
    os.makedirs(images_dest_dir / "val", exist_ok=True)
    os.makedirs(masks_dest_dir / "val", exist_ok=True)
    logger.info("Directories created")

    # Copy and resize train images and masks
    for img in tqdm(train_images, desc="Copying train images"):
        img_path = images_dir / "train" / img
        mask_path = masks_dir / "train" / img.replace(".png", "_mask.png")

        img_dest_path = images_dest_dir / "train" / img
        mask_dest_path = masks_dest_dir / "train" / img

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
    logger.info("Train images and masks resized and copied")

    # Copy and resize val images and masks
    for img in tqdm(val_images, desc="Copying val images"):
        img_path = images_dir / "val" / img
        mask_path = masks_dir / "val" / img.replace(".png", "_mask.png")

        img_dest_path = images_dest_dir / "val" / img
        mask_dest_path = masks_dest_dir / "val" / img

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
    logger.info("Val images and masks resized and copied")


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument(
        "--dir_name",
        type=str,
        default="spacecrafts",
        required=True,
        help="Name of the spacecrafts directory",
    )
    args = argparser.parse_args()

    main(args.dir_name)
