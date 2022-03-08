import os
import sys
import argparse
import numpy as np
from metrics import *
import tensorflow as tf
from dataset import get_test_dataloader
from loss import focal_loss
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from utils import show_predictions, get_config_yaml, set_gpu, create_paths

# Parsing variable
# ----------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--gpu")
parser.add_argument("--dataset_dir")
parser.add_argument("--model_name")
parser.add_argument("--load_model_name")
parser.add_argument("--single_image", type=bool)
parser.add_argument("--index", type=int)
args = parser.parse_args()

if args.single_image == 'True':
    args.single_image = True
else:
    args.single_image = False

# Set up test configaration
# ----------------------------------------------------------------------------------------------

config = get_config_yaml('config.yaml', vars(args))
create_paths(config, True)

# Setup test strategy Muli-GPU or single-GPU
# ----------------------------------------------------------------------------------------------

#strategy = set_gpu(config['gpu'])

# Dataset
# ----------------------------------------------------------------------------------------------

test_dataset = get_test_dataloader(config)


# Load Model
# ----------------------------------------------------------------------------------------------

"""load model from load_model_dir, load_model_name & model_name
   model_name is included inside the load_model_dir"""

print("Loading model {} from {}".format(config['load_model_name'], config['load_model_dir']))
#with strategy.scope():
model = load_model(os.path.join(config['load_model_dir'], config['load_model_name']), compile = False)

# Prediction Plot
# ----------------------------------------------------------------------------------------------

show_predictions(test_dataset, model, config)