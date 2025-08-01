import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging():
    """
    Configures the logging system for the application.

    - Level: DEBUG
    - Format: %(asctime)s - %(name)s - %(levelname)s - %(message)s
    - Output: Console and a rotating file in the 'logs' folder.
    - File rotation: The log file will rotate when it reaches 20MB,
      keeping up to 5 backup files.
    """
    # --- Log Directory ---
    # Created at the root of the project
    log_directory = Path(__file__).resolve().parent.parent.parent.parent / "logs"
    log_directory.mkdir(exist_ok=True)
    log_file_path = log_directory / "ecommerce_agent.log"

    # --- Formatter ---
    log_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # --- Root Logger ---
    # Get the root logger to configure it.
    # All loggers created with logging.getLogger(__name__) will inherit this configuration.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # Lowest level to capture everything

    # --- Rotating File Handler ---
    # Limit of 20 MB per file, with 5 backups.
    max_bytes = 20 * 1024 * 1024
    backup_count = 5
    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
    )
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)  # Captures everything from DEBUG upwards in the file

    # --- Console Handler ---
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)  # Shows INFO and above in console to avoid being too verbose

    # --- Add Handlers to Root Logger ---
    # Avoid adding duplicate handlers if the function is called more than once
    if not root_logger.handlers:
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

    logging.info("Logging system configured.")