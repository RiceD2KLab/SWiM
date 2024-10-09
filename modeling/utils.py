"""
This file contains utility functions that are used in the project.
"""

import os
import sys
from ml_collections import ConfigDict
from yaml import safe_load

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from consts import CONFIGS_DIR


def get_config(config_file: str) -> ConfigDict:
    """
    Load the configuration file.

    Args:
        config_file (str): The name of the configuration yaml file.

    Returns:
        ConfigDict: The configuration file.
    """
    config_path = CONFIGS_DIR / config_file
    with open(config_path, "r", encoding="utf-8") as file:
        args = safe_load(file)

    return ConfigDict(args)
