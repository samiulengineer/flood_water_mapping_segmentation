import os
import sys
import math
import glob
import argparse
import time
from loss import *
import segmentation_models as sm
from model import get_model, get_model_transfer_lr
from metrics import get_metrics
from tensorflow import keras
from utils import set_gpu, SelectCallbacks, get_config_yaml, create_paths
from dataset import get_train_val_dataloader
from tensorflow.keras.models import load_model
from tensorflow.keras import mixed_precision

tf.config.optimizer.set_jit("True")
#mixed_precision.set_global_policy('mixed_float16')


# Parsing variable
# ----------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir")
parser.add_argument("--dataset_dir")
parser.add_argument("--model_name")
parser.add_argument("--epochs")
parser.add_argument("--batch_size")
parser.add_argument("--index")
parser.add_argument("--experiment")
parser.add_argument("--patchify")
parser.add_argument("--patch_size")
parser.add_argument("--weights")
parser.add_argument("--patch_class_balance")

args = parser.parse_args()


# Set up train configaration
# ----------------------------------------------------------------------------------------------

config = get_config_yaml('config.yaml', vars(args))
create_paths(config)



# Print Experimental Setup before Training
# ----------------------------------------------------------------------------------------------

print("Model = {}".format(config['model_name']))
print("Epochs = {}".format(config['epochs']))
print("Batch Size = {}".format(config['batch_size']))
print("Preprocessed Data = {}".format(os.path.exists(config['train_dir'])))
print("Class Weigth = {}".format(str(config['weights'])))
print("Experiment = {}".format(str(config['experiment'])))



# Dataset
# ----------------------------------------------------------------------------------------------

train_dataset, val_dataset = get_train_val_dataloader(config)


# enable training strategy
metrics = list(get_metrics(config).values())
adam = keras.optimizers.Adam(learning_rate = config['learning_rate'])

# create dictionary with all custom function to pass in custom_objects
custom_obj = get_metrics(config) 
custom_obj['loss'] = focal_loss()

if (os.path.exists(os.path.join(config['load_model_dir'], config['load_model_name']))) and config['transfer_lr']:
    print("Build model for transfer learning..")
    # load model and compile
    model = load_model(os.path.join(config['load_model_dir'], config['load_model_name']), custom_objects=custom_obj, compile = True)

    model = get_model_transfer_lr(model, config['num_classes'])
    model.compile(optimizer = adam, loss = focal_loss(), metrics = metrics)

else:
    if (os.path.exists(os.path.join(config['load_model_dir'], config['load_model_name']))):
        print("Resume training from model checkpoint {}...".format(config['load_model_name']))
        # load model and compile
        model = load_model(os.path.join(config['load_model_dir'], config['load_model_name']), custom_objects=custom_obj, compile = True)

    else:
        model = get_model(config)
        model.compile(optimizer = adam, loss = focal_loss(), metrics = metrics)


# Set up Callbacks
# ----------------------------------------------------------------------------------------------

loggers = SelectCallbacks(val_dataset, model, config)
model.summary()

# fit
# ----------------------------------------------------------------------------------------------
t0 = time.time()
history = model.fit(train_dataset,
                    verbose = 1, 
                    epochs = config['epochs'],
                    validation_data = val_dataset, 
                    shuffle = False,
                    callbacks = loggers.get_callbacks(val_dataset, model),
                    )
print("training time minute: {}".format((time.time()-t0)/60))
#model.save('/content/drive/MyDrive/CSML_dataset/model/my_model.h5')
