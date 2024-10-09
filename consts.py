import os
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(__file__)).resolve()

DATA_DIR = ROOT_DIR / "data"
POSEBOWL_OBJDET_DIR = DATA_DIR / "posebowl_objdet"
POSEBOWL_SEG_DIR = DATA_DIR / "posebowl_segmented"
MODELS_DIR = ROOT_DIR / "models"

CONFIGS_DIR = ROOT_DIR / "configs"
RUNS_DIR_NAME = "runs"

YAML_TEMPLATE = """# Train, val, test sets and data directory
path: "$data_dir$" # dataset root dir
train: images/train # train dir
val: images/val # val dir
test: images/test # test dir

# Classes
names:
    0: spacecraft
"""
