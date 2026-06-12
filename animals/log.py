import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """
    Custom formatter to output structured JSON logs.
    Catches any 'metrics' passed in the 'extra' dictionary.
    """
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
        }
        
        # If we pass a 'metrics' dictionary, extract it into the JSON
        if hasattr(record, "metrics"):
            log_record["metrics"] = record.metrics
            
        return json.dumps(log_record)

def setup_logger(name="BugWorld", log_file="training_metrics.jsonl"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate logs if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # 1. Console Handler (Standard text for your terminal)
    c_handler = logging.StreamHandler()
    c_format = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    # 2. File Handler (Structured JSON for data analysis)
    f_handler = logging.FileHandler(log_file, mode='a')
    f_handler.setFormatter(JSONFormatter())
    logger.addHandler(f_handler)

    return logger