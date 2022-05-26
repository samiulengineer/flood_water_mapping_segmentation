import os
import math
import json
import pathlib
import rasterio
import numpy as np
import pandas as pd
from requests import patch
import tensorflow as tf
import albumentations as A
import earthpy.plot as ep
import rasterio
import earthpy.spatial as es
from utils import get_config_yaml
import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, Sequence
matplotlib.use('Agg')

# labels normalization values       
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



def read_img(directory, in_channels=None, label=False, patch_idx=None, height=512, width=512):
    """
    Summary:
        read image with rasterio and normalize the feature
    Arguments:
        directory (str): image path to read
        in_channels (bool): number of channels to read
        label (bool): TRUE if the given directory is mask directory otherwise False
        patch_idx (list): patch indices to read
    Return:
        numpy.array
    """

    if label:
        with rasterio.open(directory) as fmask:
            mask = fmask.read(1)
            mask[mask == 255] = 0 # convert unlabeled data
            if patch_idx:
                return mask[patch_idx[0]:patch_idx[1], patch_idx[2]:patch_idx[3]] # extract patch from original mask
            else:
                return mask
    else:
        X = np.zeros((height,width, in_channels))
        
        # read N number of channels
        for i in range(in_channels):
            tmp_ext = label_norm[i][0]
            with rasterio.open((directory+tmp_ext)) as f:
                fea = f.read(1)
            
            # normalize data
            X[:,:,i] = (fea - label_norm[i][1]) / label_norm[i][2]
        if patch_idx:
            return X[patch_idx[0]:patch_idx[1], patch_idx[2]:patch_idx[3],:] # extract patch from original features
        else:
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
    """
    Summary:
        save csv file
    Arguments:
        dictionary (dict): data as a dictionary object
        config (dict): Configuration directory
        name (str): file name to save
    Return:
        save file
    """
    df = pd.DataFrame.from_dict(dictionary)
    df.to_csv((config['dataset_dir']+name), index=False, header=True)


def data_path_split(config):
    
    """
    Summary:
        spliting data into train, test, valid
    Arguments:
        config (dict): Configuration directory
    Return:
        save file
    """

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


def class_percentage_check(label):
    
    """
    Summary:
        check class percentage of a single mask image
    Arguments:
        label (numpy.ndarray): mask image array
    Return:
        dict object holding percentage of each class
    """
    
    total_pix = label.shape[0]*label.shape[0]
    class_one = np.sum(label)
    class_zero_p = total_pix-class_one
    return {"zero_class":((class_zero_p/total_pix)*100),
            "one_class":((class_one/total_pix)*100)
    }



def save_patch_idx(path, patch_size=256, stride=8, test=None, patch_class_balance=None):
    """
    Summary:
        finding patch image indices for single image based on class percentage. work like convolutional layer
    Arguments:
        path (str): image path
        patch_size (int): size of the patch image
        stride (int): how many stride to take for each patch image
    Return:
        list holding all the patch image indices for a image
    """
    
    with rasterio.open(path) as t:
        img = t.read(1)
        img[img == 255] = 0
    
    # calculating number patch for given image
    patch_height = int((img.shape[0]-patch_size)/stride)+1 # [{(image height-patch_size)/stride}+1]
    patch_weight = int((img.shape[1]-patch_size)/stride)+1 # [{(image weight-patch_size)/stride}+1]
    
    # total patch images = patch_height * patch_weight
    patch_idx = []
    
    # image column traverse
    for i in range(patch_height):
        s_row = i*stride
        e_row = s_row+patch_size
        if e_row <= img.shape[0]:
            
            # image row traverse
            for j in range(patch_weight):
                start = (j*stride)
                end = start+patch_size
                if end <= img.shape[1]:
                    tmp = img[s_row:e_row, start:end]
                    percen = class_percentage_check(tmp) # find class percentage
                    
                    # take all patch for test images
                    if patch_class_balance or test=='test':
                        patch_idx.append([s_row, e_row, start, end])
                    
                    # store patch image indices based on class percentage
                    else:
                        if percen["one_class"]>19.0:
                            patch_idx.append([s_row, e_row, start, end])
    return  patch_idx


def write_json(target_path, target_file, data):
    """
    Summary:
        save dict object into json file
    Arguments:
        target_path (str): path to save json file
        target_file (str): file name to save
        data (dict): dictionary object holding data
    Returns:
        save json file
    """
    
    
    if not os.path.exists(target_path):
        try:
            os.makedirs(target_path)
        except Exception as e:
            print(e)
            raise
    with open(os.path.join(target_path, target_file), 'w') as f:
        json.dump(data, f)


def patch_images(data, config, name):
    """
    Summary:
        save all patch indices of all images
    Arguments:
        data: data file contain image paths
        config (dict): configuration directory
        name (str): file name to save patch indices
    Returns:
        save patch indices into file
    """
    img_dirs = []
    masks_dirs = []
    all_patch = []
    
    # loop through all images
    for i in range(len(data)):
        
        # fetching patch indices
        patches = save_patch_idx(data.masks.values[i], patch_size=config['patch_size'], stride=config['stride'], test=name.split("_")[0], patch_class_balance=config['patch_class_balance'])
        
        # generate data point for each patch image
        for patch in patches:
            img_dirs.append(data.feature_ids.values[i])
            masks_dirs.append(data.masks.values[i])
            all_patch.append(patch)
    temp = {'feature_ids': img_dirs, 'masks': masks_dirs, 'patch_idx':all_patch}
    
    # save data
    write_json(config['dataset_dir'], (name+str(config['patch_size'])+'.json'), temp)

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

    def call(self, feature_dir, label_dir, patch_idx=None):
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

        # choose random image from dataset to augment
        aug_idx = np.random.randint(0, len(feature_dir), self.aug_img_batch)
        features = []
        labels = []

        for i in aug_idx:
            if patch_idx:
                img = read_img(feature_dir[i], in_channels = self.channels, patch_idx=patch_idx[i])
                mask = read_img(label_dir[i], label=True,patch_idx=patch_idx[i])
            else:
                img = read_img(feature_dir[i], in_channels = self.channels)
                mask = read_img(label_dir[i], label=True)
            augmented = self.aug(image=img, mask=mask)
            features.append(augmented['image'])
            labels.append(augmented['mask'])
        return features, labels



# Dataloader class
# ----------------------------------------------------------------------------------------------

class MyDataset(Sequence):

    def __init__(self, img_dir, tgt_dir, in_channels, 
                 batch_size, num_class, patchify,
                 transform_fn=None, augment=None, weights=None, patch_idx=None):
        """
        Summary:
            initialize class variables
        Arguments:
            img_dir (list): all image directory
            tgt_dir (list): all mask/ label directory
            in_channels (int): number of input channels
            batch_size (int): how many data to pass in a single step
            patchify (bool): set TRUE if patchify experiment
            transform_fn (function): function to transform mask images for training
            num_class (int): number of class in mask image
            augment (object): Augment class object
            weight (list): class weight for imblance class
            patch_idx (list): list of patch indices
        Return:
            class object
        """


        self.img_dir = img_dir
        self.tgt_dir = tgt_dir
        self.patch_idx = patch_idx
        self.patchify = patchify
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


        return math.ceil(len(self.img_dir) // self.batch_size)


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
        
        
        if self.patchify:
            batch_patch = self.patch_idx[idx * self.batch_size:(idx + 1) *self.batch_size]
        
        imgs = []
        tgts = []
        for i in range(len(batch_x)):
            if self.patchify:
                imgs.append(read_img(batch_x[i], in_channels = self.in_channels, patch_idx=batch_patch[i]))
                # transform mask for model
                if self.transform_fn:
                    tgts.append(self.transform_fn(read_img(batch_y[i], label=True,patch_idx=batch_patch[i]), self.num_class))
                else:
                    tgts.append(read_img(batch_y[i], label=True,patch_idx=batch_patch[i]))
            else:
                imgs.append(read_img(batch_x[i], in_channels = self.in_channels))
                # transform mask for model
                if self.transform_fn:
                    tgts.append(self.transform_fn(read_img(batch_y[i], label=True), self.num_class))
                else:
                    tgts.append(read_img(batch_y[i], label=True))
        
        # augment data using Augment class above if augment is true
        if self.augment:
            if self.patchify:
                aug_imgs, aug_masks = self.augment.call(self.img_dir, self.tgt_dir, self.patch_idx) # augment images and mask randomly
                imgs = imgs+aug_imgs
            else:
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
        if self.patchify:
            imgs.append(read_img(self.img_dir[idx], in_channels=self.in_channels,patch_idx=self.patch_idx[idx]))
            
            # transform mask for model
            if self.transform_fn:
                tgts.append(self.transform_fn(read_img(self.tgt_dir[idx], label=True,patch_idx=self.patch_idx[idx]), self.num_class))
            else:
                tgts.append(read_img(self.tgt_dir[idx], label=True,patch_idx=self.patch_idx[idx]))
        else:
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
    
    if not (os.path.exists(config["p_train_dir"])) and config['patchify']:
        print("Saving patchify indices for train and test.....")
        data = pd.read_csv(config['train_dir'])
        
        if config["patch_class_balance"]:
            patch_images(data, config, "train_patch_WOC_")
        else:
            patch_images(data, config, "train_patch_")
        
        data = pd.read_csv(config['valid_dir'])
        
        if config["patch_class_balance"]:
            patch_images(data, config, "valid_patch_WOC_")
        else:
            patch_images(data, config, "valid_patch_")
            
        
        
    if config['patchify']:
        print("Loading Patchified features and masks directories.....")
        with open(config['p_train_dir'], 'r') as j:
            train_dir = json.loads(j.read())
        with open(config['p_valid_dir'], 'r') as j:
            valid_dir = json.loads(j.read())
        train_features = train_dir['feature_ids']
        train_masks = train_dir['masks']
        valid_features = valid_dir['feature_ids']
        valid_masks = valid_dir['masks']
        train_idx = train_dir['patch_idx']
        valid_idx = valid_dir['patch_idx']
    
    else:
        print("Loading features and masks directories.....")
        train_dir = pd.read_csv(config['train_dir'])
        valid_dir = pd.read_csv(config['valid_dir'])
        train_features = train_dir.feature_ids.values
        train_masks = train_dir.masks.values
        valid_features = valid_dir.feature_ids.values
        valid_masks = valid_dir.masks.values
        train_idx = None
        valid_idx = None

    print("train Example : {}".format(len(train_features)))
    print("valid Example : {}".format(len(valid_features)))


    # create Augment object if augment is true
    if config['augment'] and config['batch_size']>1:
        augment_obj = Augment(config['batch_size'], config['in_channels'])
        n_batch_size = config['batch_size']-augment_obj.aug_img_batch # new batch size after augment data for train
    else:
        n_batch_size = config['batch_size']
        augment_obj = None

    # class weight
    if config['weights']:
        weights=tf.constant(config['balance_weights'])
    else:
        weights = None
    
    # create dataloader object
    train_dataset = MyDataset(train_features, train_masks,
                                in_channels=config['in_channels'], patchify=config['patchify'],
                                batch_size=n_batch_size, transform_fn=transform_data, 
                                num_class=config['num_classes'], augment=augment_obj, 
                                weights=weights, patch_idx=train_idx)

    val_dataset = MyDataset(valid_features, valid_masks,
                            in_channels=config['in_channels'],patchify=config['patchify'],
                            batch_size=config['batch_size'], transform_fn=transform_data, 
                            num_class=config['num_classes'],patch_idx=valid_idx)
    
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


    if not (os.path.exists(config['test_dir'])):
        data_path_split(config)
    
    if not (os.path.exists(config["p_test_dir"])) and config['patchify']:
        print("Saving patchify indices for test.....")
        data = pd.read_csv(config['test_dir'])
        patch_images(data, config, "test_patch_")
        
    
    if config['patchify']:
        print("Loading Patchified features and masks directories.....")
        with open(config['p_test_dir'], 'r') as j:
            test_dir = json.loads(j.read())
        test_features = test_dir['feature_ids']
        test_masks = test_dir['masks']
        test_idx = test_dir['patch_idx']
    
    else:
        print("Loading features and masks directories.....")
        test_dir = pd.read_csv(config['test_dir'])
        test_features = test_dir.feature_ids.values
        test_masks = test_dir.masks.values
        test_idx = None

    print("test Example : {}".format(len(test_features)))


    test_dataset = MyDataset(test_features, test_masks,
                            in_channels=config['in_channels'],patchify=config['patchify'],
                            batch_size=config['batch_size'], transform_fn=transform_data, 
                            num_class=config['num_classes'],patch_idx=test_idx)
    
    return test_dataset



def display_all(display_list, directory, id):
    """
    Summary:
        save all images into single figure
    Arguments:
        display_list (dict): a python dictionary key is the title of the figure
        id (str) : image id in dataset
        directory (str) : path to save the plot figure
    Return:
        save images figure into directory
    """
    plt.figure(figsize=(12, 8))
    title = list(display_list.keys())

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        
        # plot dem channel using earthpy
        if title[i]=="dem":
            ax = plt.gca()
            hillshade = es.hillshade(display_list[title[i]], azimuth=180)
            ep.plot_bands(
                display_list[title[i]],
                cbar=False,
                cmap="terrain",
                title=title[i],
                ax=ax
            )
            ax.imshow(hillshade, cmap="Greys", alpha=0.5)
        
        # gray image plot vv and vh channels
        elif title[i]=="vv" or title[i]=="vh":
            plt.title(title[i])
            plt.imshow((display_list[title[i]]), cmap="gray")
            plt.axis('off')
        else:
            plt.title(title[i])
            plt.imshow((display_list[title[i]]))
            plt.axis('off')

    prediction_name = "img_id_{}.png".format(id) # create file name to save
    plt.savefig(os.path.join(directory, prediction_name), bbox_inches='tight', dpi=800)
    plt.clf()
    plt.cla()
    plt.close()

def read_img_for_display(data, directory):
    """
    Summary:
        save all images into single figure
    Arguments:
        data : data file holding images path
        directory (str) : path to save images
    Return:
        save images figure into directory
    """
    
    for i in range(len(data)):
        with rasterio.open((data.feature_ids.values[i]+"_vv.tif")) as vv:
            vv_img = vv.read(1)
        with rasterio.open((data.feature_ids.values[i]+"_vh.tif")) as vh:
            vh_img = vh.read(1)
        with rasterio.open((data.feature_ids.values[i]+"_nasadem.tif")) as dem:
            dem_img = dem.read(1)
        with rasterio.open((data.masks.values[i])) as l:
            lp_img = l.read(1)
            lp_img[lp_img==255]=0
        id = data.feature_ids.values[i].split("/")[-1]
        display_all( {"vv":vv_img,
                     "vh":vh_img,
                     "dem":dem_img,
                     "label":lp_img},
                    directory, 
                    id)



def class_balance_check(patchify, data_dir):
    """
    Summary:
        checking class percentage in full dataset
    Arguments:
        patchify (bool): TRUE if want to check class balance for patchify experiments
        data_dir (str): directory where data files save
    Return:
        class percentage
    """
    if patchify:
        with open(data_dir, 'r') as j:
            train_data = json.loads(j.read())
        labels = train_data['masks']
        patch_idx = train_data['patch_idx']
    else:
        train_data = pd.read_csv(data_dir)
        labels = train_data.masks.values
        patch_idx = None
    class_one_t = 0
    class_zero = 0
    total = 0

    for i in range(len(labels)):
        with rasterio.open(labels[i]) as l:
            mask = l.read(1)
        mask[mask == 255] = 0
        if patchify:
            idx = patch_idx[i]
            mask = mask[idx[0]:idx[1], idx[2]:idx[3]]
        total_pix = mask.shape[0]*mask.shape[1]
        total += total_pix
        class_one = np.sum(mask)
        class_one_t += class_one
        class_zero_p = total_pix-class_one
        class_zero += class_zero_p
    
    print("Water Class percentage: {}".format((class_one_t/total)*100))



if __name__=='__main__':
    
    config = get_config_yaml('config.yaml', {})

    # check class balance for patchify pass True and p_train_dir
    # check class balance for original pass False and train_dir
    class_balance_check(True, config["p_train_dir"])

    pathlib.Path((config['dataset_dir']+'display')).mkdir(parents = True, exist_ok = True)

    train_dir = pd.read_csv(config['train_dir'])
    test_dir = pd.read_csv(config['test_dir'])
    valid_dir = pd.read_csv(config['valid_dir'])
    print("Saving Figures.....")
    read_img_for_display(train_dir, (config['dataset_dir']+'display'))
    read_img_for_display(valid_dir, (config['dataset_dir']+'display'))
    read_img_for_display(test_dir, (config['dataset_dir']+'display'))
