import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import sys
import os
import random
import networkx as nx 
import logging
import argparse
import configparser

from preffect._config import configs
from preffect._logger_config import setup_loggers

from preffect.preffect_factory import(
    factory_setup,
    factory
)

def read_config_file(file_path):
    """
    Reads and parses a configuration file.

    :param file_path: The path to the configuration file (_config.py) to be read.
    :type file_path: str

    :return: A `ConfigParser` object containing the data from the configuration file.
    :rtype: configparser.ConfigParser
    """
    config = configparser.ConfigParser()
    config.read(file_path)
    return config

def str2bool(v):
    """
    Converts a string representation of truth to a boolean.

    :param v: The value to convert into a boolean. Accepts 'yes', 'true', 't', 'y', '1', and their respective
              false values 'no', 'false', 'f', 'n', '0'. Also accepts and returns a boolean value directly if
              `v` is already of type boolean.
    :type v: str or bool

    :return: The boolean value corresponding to the input.
    :rtype: bool

    :raises argparse.ArgumentTypeError: If the input string does not correspond to expected true or false values.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

    
# Main block
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Modify model and dataset parameters.')
    parser.add_argument('--config_file', type=str, help='Path to the configuration file to override default settings.')

    args, remaining_argv = parser.parse_known_args()
    if args.config_file:
        file_config = read_config_file(args.config_file)
        for section in file_config.sections():
            for key in file_config[section]:
                # Assuming all values needed are in the DEFAULT section, or adjust as needed
                configs[key] = file_config.get(section, key)


    # Assume configs is defined elsewhere
    # Merge both dictionaries for the purpose of setting up argparse arguments
    all_parameters = {**configs}

    # Dynamically add arguments based on the merged dictionaries
    for key, value in all_parameters.items():
        key_arg = f'--{key}'  # Convert key to command-line argument format
        value_type = type(value)
        
        # Special handling for boolean values and lists
        if value_type == bool:
            parser.add_argument(key_arg, type=str2bool, help='f{key} (default: {value})')
        elif value_type == list:
            parser.add_argument(key_arg, type=str, nargs='+', help=f'{key} (default: {value})')
        else:
            parser.add_argument(key_arg, type=value_type, help=f'{key} (default: {value})')

    # Parse arguments
    args = parser.parse_args(remaining_argv)

    # Update model_parameters and dataset_parameters based on passed arguments
    for key in configs.keys():
        if getattr(args, key, None) is not None:
            configs[key] = getattr(args, key)

    input_log, forward_log, inference_log, configs = factory_setup(configs=configs)
    factory(configs=configs.copy(), task=configs['task'], always_save = True)
