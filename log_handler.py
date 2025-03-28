import logging
import queue
import threading
from typing import Optional


class QueueHandler(logging.Handler):
    """Thread-safe logging handler using a queue"""

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        try:
            self.log_queue.put_nowait(record)
        except queue.Full:
            pass  # Discard log message if queue is full


class ThreadSafeLogger:
    """Thread-safe logger singleton"""

    _instance: Optional["ThreadSafeLogger"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        self.log_queue = queue.Queue(maxsize=1000)
        self.queue_handler = QueueHandler(self.log_queue)
        self.queue_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        # Main logger configuration
        self.logger = logging.getLogger("Transcriptor")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.queue_handler)

    def get_logger(self):
        return self.logger

    def get_queue(self):
        return self.log_queue
