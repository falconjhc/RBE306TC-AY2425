# # -*- coding: utf-8 -*-
# This script sets up the argument parser for training a model using a pipeline defined in the Trainer class.
# It also loads hyperparameters and penalties from the specified configuration file and initializes the model.

import argparse  # For parsing command line arguments
import os
import sys
sys.path.append('../')  # Add the parent directory to the system path to access required modules
os.chdir(sys.path[0])  # Change the current directory to the script path

# Root directory for dataset paths
dataPathRoot = '/data0/haochuan/'

import warnings  # Suppress specific warning types
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)  # Ignore future warnings

import shutup  # Suppress all other unnecessary warnings
shutup.please()

from Pipelines.Trainer import Trainer  # Import Trainer class for training
from Configurations.ConfigurationOrganization import ParameterSetting  # Import ParameterSetting class to organize parameters

# Very small constant to avoid division by zero in computations
eps = 1e-9

import importlib  # For dynamically importing configurations
from easydict import EasyDict  # Easy access to dictionary-like attributes

# Define argument parser for command line options
parser = argparse.ArgumentParser(description='Train')

# Add various arguments that the user must specify
parser.add_argument('--config', dest='config', type=str, required=True, help="Path to the configuration file")
parser.add_argument('--resumeTrain', dest='resumeTrain', type=int, required=True, help="Resume training: 0 for fresh, 1 for resuming")
parser.add_argument('--initLr', dest='initLr', type=float, required=True, help="Initial learning rate")
parser.add_argument('--batchSize', dest='batchSize', type=int, required=True, help="Batch size for training")
parser.add_argument('--epochs', dest='epochs', type=int, required=True, help="Number of epochs to train")
parser.add_argument('--skipTest', dest='skipTest', type=bool, default=False, help="Skip test step if set to True")
parser.add_argument('--encoder', dest='encoder', type=str, required=False, help="Encoder type")
parser.add_argument('--mixer', dest='mixer', type=str, required=True, help="Mixer type")
parser.add_argument('--decoder', dest='decoder', type=str, required=False, default=None, help="Decoder type, optional")
parser.add_argument('--device', dest='device', type=str, required=True, default='cpu', help="Device to use, e.g., 'cpu' or 'cuda'")
parser.add_argument('--debug', dest='debug', type=int, required=False, default=0, help="Enable debugging mode: 1 for debug, 0 for normal")
parser.add_argument('--wnet', dest='wnet', type=str, required=True, default='general', help="Type of network: 'general' or 'plain'")

if __name__ == '__main__':
    # Parse command line arguments
    args = parser.parse_args()

    # Load hyperparameters and penalties from the specified config file
    hyperParams = ParameterSetting(EasyDict(importlib.import_module('.'+args.config, package='Configurations').hyperParams), args).config 
    penalties = EasyDict(importlib.import_module('.'+args.config, package='Configurations').penalties)
    
    # Set the CUDA device environment variable
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    # Display a visual separator in the console output
    print("#####################################################")

    # Initialize the Trainer object with the loaded hyperparameters and penalties
    model = Trainer(hyperParams=hyperParams, penalties=penalties)
    
    # Start the training pipeline
    model.Pipelines()