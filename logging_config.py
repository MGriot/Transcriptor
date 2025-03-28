import logging
import os
from datetime import datetime
from log_handler import ThreadSafeLogger


def setup_logging(log_dir="logs"):
    """Setup logging with both file and queue handlers"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(
        log_dir, f"transcriptor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    # File handler setup
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    # Get the thread-safe logger instance
    logger = ThreadSafeLogger().get_logger()
    logger.addHandler(file_handler)

    return logger, ThreadSafeLogger().get_queue()
