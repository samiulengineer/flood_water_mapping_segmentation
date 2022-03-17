import os
import math
import yaml
import itertools
import gc
import numpy as np
import pathlib
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K


# Callbacks and Prediction during Training
# ----------------------------------------------------------------------------------------------
class SelectCallbacks(keras.callbacks.Callback):
    def __init__(self, val_dataset, model, config):
        """
        Summary:
            callback class for validation prediction and create the necessary callbacks objects
        Arguments:
            val_dataset (object): MyDataset class object
            model (object): keras.Model object
            config (dict): configuration dictionary
        Return:
            class object
        """
        super(keras.callbacks.Callback, self).__init__()

        self.val_dataset = val_dataset
        self.model = model
        self.config = config
        self.callbacks = []

    def lr_scheduler(self, epoch):
        """
        Summary:
            learning rate decrease according to the model performance
        Arguments:
            epoch (int): current epoch
        Return:
            learning rate
        """
        drop = 0.5
        epoch_drop = self.config['epochs'] / 8.
        lr = self.config['learning_rate'] * math.pow(drop, math.floor((1 + epoch) / epoch_drop))
        return lr

    def on_epoch_end(self, epoch, logs={}):
        """
        Summary:
            call after every epoch to predict mask
        Arguments:
            epoch (int): current epoch
        Output:
            save predict mask
        """
        if (epoch % self.config['val_plot_epoch'] == 0): # every after certain epochs the model will predict mask
            gc.collect()
            show_predictions(self.val_dataset, self.model, self.config, True)

    def get_callbacks(self, val_dataset, model):
        """
        Summary:
            creating callbacks based on configuration
        Arguments:
            val_dataset (object): MyDataset class object
            model (object): keras.Model class object
        Return:
            list of callbacks
        """
        if self.config['csv']:
            self.callbacks.append(keras.callbacks.CSVLogger(os.path.join(self.config['csv_log_dir'], self.config['csv_log_name']), separator = ",", append = False))
        
        if self.config['checkpoint']:
            self.callbacks.append(keras.callbacks.ModelCheckpoint(os.path.join(self.config['checkpoint_dir'], self.config['checkpoint_name']), save_best_only = True))
        
        if self.config['tensorboard']:
            self.callbacks.append(keras.callbacks.TensorBoard(log_dir = os.path.join(self.config['tensorboard_log_dir'], self.config['tensorboard_log_name'])))
        
        if self.config['lr']:
            self.callbacks.append(keras.callbacks.LearningRateScheduler(schedule = self.lr_scheduler))
        
        if self.config['early_stop']:
            self.callbacks.append(keras.callbacks.EarlyStopping(monitor = 'my_mean_iou', patience = self.config['patience']))
        
        if self.config['val_pred_plot']:
            self.callbacks.append(SelectCallbacks(val_dataset, model, self.config))
        
        return self.callbacks

# Prepare masks
# ----------------------------------------------------------------------------------------------
def create_mask(mask, pred_mask):
    """
    Summary:
        apply argmax on mask and pred_mask class dimension
    Arguments:
        mask (ndarray): image labels/ masks
        pred_mask (ndarray): prediction labels/ masks
    Return:
        return mask and pred_mask after argmax
    """
    mask = np.argmax(mask, axis = 3)
    pred_mask = np.argmax(pred_mask, axis = 3)
    return mask, pred_mask

# Sub-ploting and save
# ----------------------------------------------------------------------------------------------
def display(display_list, idx, directory, score, exp):
    """
    Summary:
        save all images into single figure
    Arguments:
        display_list (dict): a python dictionary key is the title of the figure
        idx (int) : image index in dataset object
        directory (str) : path to save the plot figure
        score (float) : accuracy of the predicted mask
    Return:
        save images figure into directory
    """
    plt.figure(figsize=(12, 8))
    title = list(display_list.keys())

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow((display_list[title[i]]))
        plt.axis('off')

    prediction_name = "img_ex_{}_{}_acc_{:.4f}.png".format(exp, idx, score) # create file name to save
    return plt.savefig(os.path.join(directory, prediction_name), bbox_inches='tight')


# Save all plot figures
# ----------------------------------------------------------------------------------------------
def show_predictions(dataset, model, config, val=False):
    """
    Summary: 
        save image/images with their mask, pred_mask and accuracy
    Arguments:
        dataset (object): MyDataset class object
        model (object): keras.Model class object
        config (dict): configuration dictionary
    Output:
        save predicted image/images
    """

    if val:
        directory = config['prediction_val_dir']
    else:
        directory = config['prediction_test_dir']

    # save single image after prediction from dataset
    if config['single_image']:
        feature, mask, idx = dataset.get_random_data(config['index'])
        data = [(feature, mask)]
    else:
        data = dataset
        idx = 0

    for feature, mask in data: # save all image prediction in the dataset
        prediction = model.predict_on_batch(feature)
        mask, pred_mask = create_mask(mask, prediction)
        for i in range(len(feature)): # save single image prediction in the batch
            m = keras.metrics.MeanIoU(num_classes=6)
            m.update_state(mask[i], pred_mask[i])
            score = m.result().numpy()
            display({"Feature": feature[i],
                      "Mask": mask[i],
                      "Prediction (MeanIOU_{:.4f})".format(score): pred_mask[i]
                      }, idx, directory, score, config['experiment'])
            idx += 1

# GPU setting
# ----------------------------------------------------------------------------------------------
def set_gpu(gpus):
    """
    Summary:
        setting multi-GPUs or single-GPU strategy for training
    Arguments:
        gpus (str): comma separated str variable i.e. "0,1,2"
    Return:
        gpu strategy object
    """
    gpus = gpus.split(",")
    if len(gpus)>1:
        print("MirroredStrategy Enable")
        GPUS = []
        for i in range(len(gpus)):
            GPUS.append("GPU:{}".format(gpus[i]))
        strategy = tf.distribute.MirroredStrategy(GPUS)
    else:
        print("OneDeviceStrategy Enable")
        GPUS = []
        for i in range(len(gpus)):
            GPUS.append("GPU:{}".format(gpus[i]))
        strategy = tf.distribute.OneDeviceStrategy(GPUS[0])
    print('Number of devices: %d' % strategy.num_replicas_in_sync)
    
    return strategy

# Model Output Path
# ----------------------------------------------------------------------------------------------
def create_paths(config, test=False):
    """
    Summary:
        creating paths for train and test if not exists
    Arguments:
        config (dict): configuration dictionary
        test (bool): boolean variable for test directory create
    Return:
        create directories
    """
    if test:
        pathlib.Path(config['prediction_test_dir']).mkdir(parents = True, exist_ok = True)
    else:
        pathlib.Path(config['csv_log_dir']).mkdir(parents = True, exist_ok = True)
        pathlib.Path(config['tensorboard_log_dir']).mkdir(parents = True, exist_ok = True)
        pathlib.Path(config['checkpoint_dir']).mkdir(parents = True, exist_ok = True)
        pathlib.Path(config['prediction_val_dir']).mkdir(parents = True, exist_ok = True)

# Create config path
# ----------------------------------------------------------------------------------------------
def get_config_yaml(path, args):
    """
    Summary:
        parsing the config.yaml file and re organize some variables
    Arguments:
        path (str): config.yaml file directory
        args (dict): dictionary of passing arguments
    Return:
        a dictonary
    """
    with open(path, "r") as f:
      config = yaml.safe_load(f)
    
    # Replace default values with passing values
    for key in args.keys():
        if args[key] != None:
            config[key] = args[key]
            
    config['height'] = config['patch_size']
    config['width'] = config['patch_size']
    
    # Merge paths
    config['train_dir'] = config['dataset_dir']+config['train_dir']
    config['valid_dir'] = config['dataset_dir']+config['valid_dir']
    config['test_dir'] = config['dataset_dir']+config['test_dir']
    
    config['p_train_dir'] = config['dataset_dir']+config['p_train_dir']
    config['p_valid_dir'] = config['dataset_dir']+config['p_valid_dir']
    config['p_test_dir'] = config['dataset_dir']+config['p_test_dir']
    
    # Create Callbacks paths
    config['tensorboard_log_name'] = "{}_ex_{}_epochs_{}_{}".format(config['model_name'],config['experiment'],config['epochs'],datetime.now().strftime("%d-%b-%y"))
    config['tensorboard_log_dir'] = config['root_dir']+'/logs/'+config['model_name']+'/'

    config['csv_log_name'] = "{}_ex_{}_epochs_{}_{}.csv".format(config['model_name'],config['experiment'],config['epochs'],datetime.now().strftime("%d-%b-%y"))
    config['csv_log_dir'] = config['root_dir']+'/csv_logger/'+config['model_name']+'/'

    config['checkpoint_name'] = "{}_ex_{}_epochs_{}_{}.hdf5".format(config['model_name'],config['experiment'],config['epochs'],datetime.now().strftime("%d-%b-%y"))
    config['checkpoint_dir'] = config['root_dir']+'/model/'+config['model_name']+'/'

    # Create save model directory
    if config['load_model_dir']=='None':
        config['load_model_dir'] = config['root_dir']+'/model/'+config['model_name']+'/'
    
    # Create Evaluation directory
    config['prediction_test_dir'] = config['root_dir']+'/prediction/'+config['model_name']+'/test/'
    config['prediction_val_dir'] = config['root_dir']+'/prediction/'+config['model_name']+'/validation/'

    return config
