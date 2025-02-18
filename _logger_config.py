import os
import logging
from datetime import datetime
import time
import logging


def setup_loggers(configs):
    """
    Sets up the loggers for the application using the provided configuration.

    Args:
        configs (dict): Configuration dictionary that includes settings such as the log directory path.

    Returns:
        tuple: A tuple containing references to the input, forward, and inference loggers.
    """
    log_directory = configs['log_path']
    # Ensure the log directory exists
    os.makedirs(log_directory, exist_ok=True)

    # Setup the 'input' logger
    input_log = logging.getLogger('input')
    input_log.setLevel(logging.INFO)
    input_file_handler = logging.FileHandler(os.path.join(log_directory, "input_preffect.log"), mode="w")
    input_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    input_file_handler.setFormatter(input_formatter)
    input_log.addHandler(input_file_handler)
    input_log.info(datetime.fromtimestamp(time.time()))

    # Setup the 'forward' logger
    forward_log = logging.getLogger('forward')
    forward_log.setLevel(logging.INFO)
    forward_file_handler = logging.FileHandler(os.path.join(log_directory, "forward_preffect.log"), mode="w")
    forward_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    forward_file_handler.setFormatter(forward_formatter)
    forward_log.addHandler(forward_file_handler)
    forward_log.info(datetime.fromtimestamp(time.time()))

    # Setup the 'inference' logger
    inference_log = logging.getLogger('inference')
    inference_log.setLevel(logging.INFO)
    inference_file_handler = logging.FileHandler(os.path.join(log_directory, "inference_preffect.log"), mode="w")
    inference_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    inference_file_handler.setFormatter(inference_formatter)
    inference_log.addHandler(inference_file_handler)
    inference_log.info(datetime.fromtimestamp(time.time()))
    # Optionally, return the loggers if you need references to them immediately
    return input_log, forward_log, inference_log
