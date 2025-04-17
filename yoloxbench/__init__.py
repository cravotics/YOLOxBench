"""Top‑level package for yoloxbench."""
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("yoloxbench")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"