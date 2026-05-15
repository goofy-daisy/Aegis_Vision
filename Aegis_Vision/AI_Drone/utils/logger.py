"""Centralized logging utility for AegisVision.

Provides a consistent logging interface across all modules with both
console and file handlers. All modules should use get_logger(__name__)
to obtain a configured logger instance.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def get_logger(name: str, log_dir: Optional[Path] = None) -> logging.Logger:
    """Create and configure a logger with file and console handlers.

    Args:
        name: The name of the logger, typically __name__ from the calling module.
        log_dir: Directory for log files. If None, uses project_root/logs.

    Returns:
        A configured logging.Logger instance.
    """
    # Create logger instance
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger
    
    # Determine log directory relative to project root
    if log_dir is None:
        # Traverse up to find project root (where configs/ exists)
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        log_dir = project_root / "logs"
    
    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set logger level to DEBUG to capture all messages
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    # Format: [TIMESTAMP] [LEVEL] [MODULE] message
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler - INFO level and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler - DEBUG level and above, rotating with 5MB max size, 3 backups
    log_file = log_dir / f"{name.replace('.', '_')}.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
