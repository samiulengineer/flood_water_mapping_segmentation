import os 
import cv2
import math
import glob
import shutil
import random
import rasterio
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import albumentations as A
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, Sequence

# unpack labels        
label_norm = {0:["_vv.tif", -17.54, 5.15],
                1:["_vh.tif",-10.68, 4.62],
                2:["_nasadem.tif",166.47, 178.47],
                3:["_jrc-gsw-change.tif", 238.76, 5.15],
                4:["_jrc-gsw-extent.tif", 2.15, 22.71],
                5:["_jrc-gsw-occurrence.tif", 6.50, 29.06],
                6:["_jrc-gsw-recurrence.tif", 10.04, 33.21],
                7:["_jrc-gsw-seasonality.tif", 2.60, 22.79],
                8:["_jrc-gsw-transitions.tif", 0.55, 1.94]}


def transform_data(label, num_classes):
    """
    Summary:
        transform label/mask into one hot matrix and return
    Arguments:
        label (arr): label/mask
        num_classes (int): number of class in label/mask
    Return:
        one hot label matrix
    """


    return to_categorical(label, num_classes = num_classes)



def read_img(directory, in_channels=None, label=False):
    """
    Summary:
        read image with opencv and normalize the feature
    Arguments:
        directory (str): image path to read
        rgb (bool): convert BGR to RGB image as opencv read in BGR format
    Return:
        numpy.array
    """

    if label:
        with rasterio.open(directory) as fmask:
            mask = fmask.read(1)
            mask[mask == 255] = 0
            return mask
    else:
        X = np.zeros((512,512, in_channels))
        for i in range(in_channels):
            tmp_ext = label_norm[i][0]
            #print("{}/{}{}".format(train_dir,path,tmp_ext))
            with rasterio.open((directory+tmp_ext)) as f:
                fea = f.read(1)
            X[:,:,i] = (fea - label_norm[i][1]) / label_norm[i][2]
        return X


def data_split(images, masks, config):
    """
    Summary:
        split dataset into train, valid and test
    Arguments:
        images (list): all image directory list
        masks (list): all mask directory
        config (dict): Configuration directory
    Return:
        return the split data.
    """


    x_train, x_rem, y_train, y_rem = train_test_split(images, masks, train_size = config['train_size'], random_state=42)
    x_valid, x_test, y_valid, y_test = train_test_split(x_rem, y_rem, test_size = 0.5, random_state=42)
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def save_csv(dictionary, config, name):
    df = pd.DataFrame.from_dict(dictionary)
    df.to_csv((config['dataset_dir']+name), index=False, header=True)


def data_path_split(config):

    paths = pd.read_csv((config['dataset_dir']+"flood-training-metadata.csv"))
    paths = paths.drop_duplicates('chip_id').reset_index(drop=True)

    ids = list(paths.chip_id.values)

    masks = []
    images = []
    for i in range(len(ids)):
        masks.append(config['dataset_dir']+"train_labels/"+ids[i]+".tif")
        images.append(config['dataset_dir']+"train_features/"+ids[i])
    
    x_train, y_train, x_valid, y_valid, x_test, y_test = data_split(images, masks, config)
    
    train = {'feature_ids': x_train, 'masks': y_train}
    valid = {'feature_ids': x_valid, 'masks': y_valid}
    test = {'feature_ids': x_test, 'masks': y_test}

    save_csv(train, config, "train.csv")
    save_csv(valid, config, "valid.csv")
    save_csv(test, config, "test.csv")



# Data Augment class
# ----------------------------------------------------------------------------------------------
class Augment:
    def __init__(self, batch_size, channels, ratio=0.3, seed=42):
        super().__init__()
        """
        Summary:
            initialize class variables
        Arguments:
            batch_size (int): how many data to pass in a single step
            ratio (float): percentage of augment data in a single batch
            seed (int): both use the same seed, so they'll make the same random changes.
        Return:
            class object
        """


        self.ratio=ratio
        self.channels= channels
        self.aug_img_batch = math.ceil(batch_size*ratio)
        self.aug = A.Compose([
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Blur(p=0.5),])

    def call(self, feature_dir, label_dir):
        """
        Summary:
            randomly select a directory and augment data 
            from that specific image and mask
        Arguments:
            feature_dir (list): all train image directory list
            label_dir (list): all train mask directory list
        Return:
            augmented image and mask
        """


        aug_idx = np.random.randint(0, len(feature_dir), self.aug_img_batch)
        features = []
        labels = []

        for i in aug_idx:
            img = read_img(feature_dir[i], in_channels = self.channels)
            mask = read_img(label_dir[i], label=True)
            augmented = self.aug(image=img, mask=mask)
            features.append(augmented['image'])
            labels.append(augmented['mask'])
        return features, labels



# Dataloader class
# ----------------------------------------------------------------------------------------------

class MyDataset(Sequence):

    def __init__(self, img_dir, tgt_dir, in_channels, batch_size, num_class, transform_fn=None, augment=None, weights=None,):
        """
        Summary:
            initialize class variables
        Arguments:
            img_dir (list): all image directory
            tgt_dir (list): all mask/ label directory
            batch_size (int): how many data to pass in a single step
            transform_fn (function): function to transform mask images for training
            num_class (int): number of class in mask image
            augment (object): Augment class object
            weight (list): class weight for imblance class
        Return:
            class object
        """


        self.img_dir = img_dir
        self.tgt_dir = tgt_dir
        self.in_channels = in_channels
        self.transform_fn = transform_fn
        self.batch_size = batch_size
        self.num_class = num_class
        self.augment = augment
        self.weights = weights



    def __len__(self):
        """
        return total number of batch to travel full dataset
        """


        return math.ceil(len(self.img_dir) / self.batch_size)


    def __getitem__(self, idx):
        """
        Summary:
            create a single batch for training
        Arguments:
            idx (int): sequential batch number
        Return:
            images and masks as numpy array for a single batch
        """


        # get index for single batch
        batch_x = self.img_dir[idx * self.batch_size:(idx + 1) *self.batch_size]
        batch_y = self.tgt_dir[idx * self.batch_size:(idx + 1) *self.batch_size]
        imgs = []
        tgts = []
        for i in range(len(batch_x)):
          imgs.append(read_img(batch_x[i], in_channels = self.in_channels))
          # transform mask for model
          if self.transform_fn:
            tgts.append(self.transform_fn(read_img(batch_y[i], label=True), self.num_class))
          else:
            tgts.append(read_img(batch_y[i], label=True))
        
        # augment data using Augment class above if augment is true
        if self.augment:
            aug_imgs, aug_masks = self.augment.call(self.img_dir, self.tgt_dir) # augment images and mask randomly
            imgs = imgs+aug_imgs

            # transform mask for model
            if self.transform_fn:
                for i in range(len(aug_masks)):
                    tgts.append(self.transform_fn(aug_masks[i], self.num_class))
            else:
                tgts = tgts+aug_masks


        tgts = np.array(tgts)
        imgs = np.array(imgs)        

        if self.weights != None:

            class_weights = tf.constant(self.weights)
            class_weights = class_weights/tf.reduce_sum(class_weights)
            y_weights = tf.gather(class_weights, indices=tf.cast(tgts, tf.int32))#([self.paths[i] for i in indexes])

            return tf.convert_to_tensor(imgs), y_weights

        return tf.convert_to_tensor(imgs), tf.convert_to_tensor(tgts)
    

    def get_random_data(self, idx=-1):
        """
        Summary:
            randomly chose an image and mask or the given index image and mask
        Arguments:
            idx (int): specific image index default -1 for random
        Return:
            image and mask as numpy array
        """



        if idx!=-1:
            idx = idx
        else:
            idx = np.random.randint(0, len(self.img_dir))
        
        imgs = []
        tgts = []
        imgs.append(read_img(self.img_dir[idx], in_channels=self.in_channels))
        
        # transform mask for model
        if self.transform_fn:
            tgts.append(self.transform_fn(read_img(self.tgt_dir[idx], label=True), self.num_class))
        else:
            tgts.append(read_img(self.tgt_dir[idx], label=True))

        return tf.convert_to_tensor(imgs), tf.convert_to_tensor(tgts), idx


def get_train_val_dataloader(config):
    """
    Summary:
        read train and valid image and mask directory and return dataloader
    Arguments:
        config (dict): Configuration directory
    Return:
        train and valid dataloader
    """


    if not (os.path.exists(config['train_dir'])):
        data_path_split(config)
    else:
        print("Loading features and masks directories.....")
    
    train_dir = pd.read_csv(config['train_dir'])
    valid_dir = pd.read_csv(config['valid_dir'])

    print("train Example : {}".format(len(train_dir)))
    print("valid Example : {}".format(len(valid_dir)))


    # create Augment object if augment is true
    if config['augment'] and config['batch_size']>1:
        augment_obj = Augment(config['batch_size'], config['in_channels'])
        n_batch_size = config['batch_size']-augment_obj.aug_img_batch # new batch size after augment data for train
    else:
        n_batch_size = config['batch_size']
        augment_obj = None

    # create dataloader object
    weights=tf.constant([1.0, 8.0])
    train_dataset = MyDataset(train_dir.feature_ids.values, train_dir.masks.values,
                                in_channels=config['in_channels'],
                                batch_size=n_batch_size, transform_fn=transform_data, 
                                num_class=config['num_classes'], augment=augment_obj, weights=weights)

    val_dataset = MyDataset(valid_dir.feature_ids.values, valid_dir.masks.values, 
                            in_channels=config['in_channels'],
                            batch_size=config['batch_size'], transform_fn=transform_data, 
                            num_class=config['num_classes'], weights=weights)
    
    return train_dataset, val_dataset


def get_test_dataloader(config):
    """
    Summary:
        read test image and mask directory and return dataloader
    Arguments:
        config (dict): Configuration directory
    Return:
        test dataloader
    """


    if not (os.path.exists(config['train_dir'])):
        data_path_split(config)
    else:
        print("Loading test data directories.....")
    
    test_dir = pd.read_csv(config['test_dir'])

    
    print("test Example : {}".format(len(test_dir)))

    # create dataloader object
    test_dataset = MyDataset(test_dir.feature_ids.values, test_dir.masks.values, 
                            in_channels=config['in_channels'],
                            batch_size=config['batch_size'], transform_fn=transform_data, 
                            num_class=config['num_classes'], weights=tf.constant([1.0, 8.0]))
    
    return test_dataset