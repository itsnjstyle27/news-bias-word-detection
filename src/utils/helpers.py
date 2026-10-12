import os
import random
import logging
import yaml
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fix random seeds across python, numpy, and random for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML project configuration safely."""
    if not os.path.isabs(config_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(base_dir, config_path)
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(name: str = "bias_detector", log_file: str = None) -> logging.Logger:
    """Create a configured logger with standard stream formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger


def format_metrics_table(headers: list, rows: list) -> str:
    """Format tabular data into an aligned Markdown-compatible text table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(val)))

    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep_line = "-|-".join("-" * w for w in col_widths)
    row_lines = [
        " | ".join(str(val).ljust(w) for val, w in zip(row, col_widths))
        for row in rows
    ]
    return f"| {header_line} |\n| {sep_line} |\n" + "\n".join(f"| {r} |" for r in row_lines)
