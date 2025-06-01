"""Utility helpers shared across modules."""
import logging
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "pipeline.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_logger(name: str):
    """Return a configured logger."""
    return logging.getLogger(name)