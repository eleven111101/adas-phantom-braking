import logging
import os

def get_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("ADAS_PIPELINE")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(file_handler)
    return logger