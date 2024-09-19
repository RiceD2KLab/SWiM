import os
import sys
from typing import Tuple, List
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from PIL import Image
import pybboxes as pbx
from loguru import logger
from tqdm import tqdm
from ultralytics import SAM

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from paths import POSEBOWL_OBJDET_DIR, DATA_DIR, MODELS_DIR


def setup(input_dir: str, output_dir) -> Tuple[List, List, List]:
    """
    Create the necessary directories and return the ids of the images in the train, test, and val splits.

    Args:
        input_dir (str): The path to the input directory.
        output_dir (str): The path to the output directory.

    Returns:
        Tuple[List, List, List]: The ids of the images in the train, test, and val splits.
    """
    os.makedirs(output_dir / "images" / "train", exist_ok=True)
    os.makedirs(output_dir / "images" / "test", exist_ok=True)
    os.makedirs(output_dir / "images" / "val", exist_ok=True)

    os.makedirs(output_dir / "masks" / "train", exist_ok=True)
    os.makedirs(output_dir / "masks" / "test", exist_ok=True)
    os.makedirs(output_dir / "masks" / "val", exist_ok=True)

    os.makedirs(output_dir / "bboxes" / "train", exist_ok=True)
    os.makedirs(output_dir / "bboxes" / "test", exist_ok=True)
    os.makedirs(output_dir / "bboxes" / "val", exist_ok=True)

    train_ids = list(
        map(
            lambda img: os.path.splitext(img)[0],
            os.listdir(input_dir / "images" / "train"),
        )
    )
    test_ids = list(
        map(
            lambda img: os.path.splitext(img)[0],
            os.listdir(input_dir / "images" / "test"),
        )
    )
    val_ids = list(
        map(
            lambda img: os.path.splitext(img)[0],
            os.listdir(input_dir / "images" / "val"),
        )
    )

    return train_ids, test_ids, val_ids


def get_dfs(
    output_dir: str, train_ids: List, val_ids: List, test_ids: List
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create the train, test, and val DataFrames.

    Args:
        train_ids (List): The ids of the images in the train split.
        val_ids (List): The ids of the images in the val split.
        test_ids (List): The ids of the images in the test split.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The train, test, and val DataFrames.
    """

    # Check if the train, test, and val DataFrames already exist
    if (
        os.path.exists(output_dir / "train.csv")
        and os.path.exists(output_dir / "test.csv")
        and os.path.exists(output_dir / "val.csv")
    ):
        train_df = pd.read_csv(output_dir / "train.csv", header=0)
        test_df = pd.read_csv(output_dir / "test.csv", header=0)
        val_df = pd.read_csv(output_dir / "val.csv", header=0)
    else:
        train_df = pd.DataFrame(
            {
                "ids": train_ids,
                "split": "train",
                "processed": pd.Series(np.zeros(len(train_ids), dtype=bool)),
            },
        )
        test_df = pd.DataFrame(
            {
                "ids": test_ids,
                "split": "test",
                "processed": pd.Series(np.zeros(len(test_ids), dtype=bool)),
            },
        )
        val_df = pd.DataFrame(
            {
                "ids": val_ids,
                "split": "val",
                "processed": pd.Series(np.zeros(len(val_ids), dtype=bool)),
            },
        )

    train_df.set_index("ids", inplace=True)
    test_df.set_index("ids", inplace=True)
    val_df.set_index("ids", inplace=True)

    train_df.to_csv(output_dir / "train.csv")
    test_df.to_csv(output_dir / "test.csv")
    val_df.to_csv(output_dir / "val.csv")

    return train_df, test_df, val_df


def get_bbox(img: Image, bbox_path: Path) -> List:
    """
    Get the bounding box of the object in the image.

    Args:
        img (Image): The image.
        bbox_path (Path): The path to the bounding box file.

    Returns:
        List: The bbox of the object in the image.
    """
    bbox = None
    with open(bbox_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 1, f"Expected 1 bbox per image, got {len(lines)}"
        _, x, y, w, h = lines[0].strip().split(" ")
        x, y, w, h = map(float, [x, y, w, h])
        x_tl, y_tl, x_br, y_br = pbx.convert_bbox(
            (x, y, w, h), from_type="yolo", to_type="voc", image_size=img.size
        )
        bbox = [0, x_tl, y_tl, x_br, y_br]

    return bbox


def get_mask(img: Image, bbox: List, sam: SAM) -> Image:
    """
    Generate the mask of the object in the image using SAM2.

    Args:
        img (Image): The image.
        bbox (List): The bbox of the object in the image.
        sam (SAM): The SAM2 Model object.

    Returns:
        Image: The mask of the object in the image.
    """
    results = sam(img, bboxes=bbox[1:], verbose=False)
    mask = results[0].masks.data.cpu().numpy().astype(np.uint8).squeeze(0) * 255

    return mask


def process_split(
    split_df: pd.DataFrame, input_dir: Path, output_dir: Path, sam: SAM
) -> None:
    """
    Process the images in the split.

    Args:
        split_df (pd.DataFrame): The DataFrame containing the split images.
        input_dir (Path): The path to the input directory.
        output_dir (Path): The path to the output directory.
        sam (SAM): The SAM2 Model object.

    Returns:
        None
    """
    for idx, row in tqdm(split_df.iterrows(), total=len(split_df)):
        img_id = idx
        split = row["split"]

        if split_df.at[idx, "processed"]:
            continue

        img_path = input_dir / "images" / split / f"{img_id}.jpg"
        bbox_path = input_dir / "labels" / split / f"{img_id}.txt"

        img = Image.open(img_path)
        bbox = get_bbox(img, bbox_path)
        mask = Image.fromarray(get_mask(img, bbox, sam))

        # Save the mask and the image
        img.save(output_dir / "images" / split / f"{img_id}.png")
        mask.save(output_dir / "masks" / split / f"{img_id}.png")

        # Save the bounding box coordinates
        with open(
            output_dir / "bboxes" / split / f"{img_id}.txt", "w", encoding="utf-8"
        ) as f:
            f.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")

        split_df.at[idx, "processed"] = True
        split_df.to_csv(output_dir / f"{split}.csv")


def main(split: str) -> None:
    input_dir = POSEBOWL_OBJDET_DIR
    output_dir = DATA_DIR / "posebowl_segmented"

    train_ids, test_ids, val_ids = setup(input_dir, output_dir)

    logger.debug(f"Found {len(train_ids)} train images")
    logger.debug(f"Found {len(test_ids)} train images")
    logger.debug(f"Found {len(val_ids)} train images")

    train_df, test_df, val_df = get_dfs(output_dir, train_ids, val_ids, test_ids)

    sam = SAM(MODELS_DIR / "sam2_b.pt")
    logger.info("Loaded SAM2 model")
    logger.debug(sam.info())

    logger.info(f"Processing {split} images")

    if split == "train":
        process_split(train_df, input_dir, output_dir, sam)
    elif split == "test":
        process_split(test_df, input_dir, output_dir, sam)
    elif split == "val":
        process_split(val_df, input_dir, output_dir, sam)
    else:
        raise ValueError("Invalid split")


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument(
        "--split",
        type=str,
        default="val",
        help="The split to process (train, test, val)",
    )
    args = argparser.parse_args()
    main(args.split)
