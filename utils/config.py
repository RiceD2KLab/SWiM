"""
This file contains utility functions that are used in the project.
"""

from pathlib import Path
from ml_collections import ConfigDict
from yaml import safe_load


def get_config(config_path: Path) -> ConfigDict:
    """
    Load the configuration file.

    Args:
        config_file (Path): The path to the configuration file.

    Returns:
        ConfigDict: The configuration file.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        args = safe_load(file)

    return ConfigDict(args)
