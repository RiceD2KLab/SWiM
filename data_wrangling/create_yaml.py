"""
    This script creates YAML file for YOLO training.

    Args:
        data_dir (str): The name of the data directory.
    
    Results:
        YAML file created in the data directory.
"""

from argparse import ArgumentParser
import os
import sys

from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from consts import DATA_DIR, YAML_TEMPLATE


def main(data_dir_name: str) -> None:
    """
    Create a YAML file with the data directory name.

    Args:
        data_dir_name (str): The name of the data directory.
    """

    data_dir = DATA_DIR / data_dir_name
    images_dir = data_dir / "images"
    # masks_dir = data_dir / "masks"

    if not data_dir.exists():
        raise FileNotFoundError(f"Directory {data_dir} does not exist.")
    if not images_dir.exists():
        raise FileNotFoundError(f"Directory {images_dir} does not exist.")
    # if not masks_dir.exists():
    #     raise FileNotFoundError(f"Directory {masks_dir} does not exist.")
    logger.info(f"Creating YAML file in {data_dir.resolve()}.")

    yaml_content = YAML_TEMPLATE.replace("$data_dir$", str(data_dir.resolve())).replace(
        "\\", "/"
    )
    with open(data_dir / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    logger.info("YAML file created.")


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument(
        "--data_dir",
        type=str,
        help="The directory with images/ and labels/ subdirectories.",
    )
    args = argparser.parse_args()
    main(args.data_dir)
