from importlib.metadata import version

from . import utils
from .analyzer import analyze_mask

__version__ = version("grainmorph")
__all__ = ["analyze_mask", "utils"]
