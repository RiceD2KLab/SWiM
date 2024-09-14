import os
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(__file__))

DATA_DIR = ROOT_DIR / "data"
POSEBOWL_OBJDET_DIR = DATA_DIR / "posebowl_objdet"
MODELS_DIR = ROOT_DIR / "models"
