import pathlib
import importlib.util
import sys
import traceback
from pathlib import Path


root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "nodes"))

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
