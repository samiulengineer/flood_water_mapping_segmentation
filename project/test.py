import os
import glob
import argparse
import json
import numpy as np
import pandas as pd
from metrics import *
import matplotlib.pyplot as plt
import earthpy.plot as ep
import earthpy.spatial as es
from dataset import get_test_dataloader, read_img, transform_data
from loss import *
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils import show_predictions, get_config_yaml, create_paths, display

#model = load_model(os.path.join('/home/mdsamiul/github_project/flood_water_mapping_segmentation/model/mnet/mnet_ex_patchify_n3_256_epochs_300_15-Apr-22.hdf5'), compile = False)
# Parsing variable
# ----------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir")
parser.add_argument("--model_name")
parser.add_argument("--load_model_name")
parser.add_argument("--plot_single", type=bool)
parser.add_argument("--index", type=int)
parser.add_argument("--experiment")
parser.add_argument("--patchify")
parser.add_argument("--patch_size")
parser.add_argument("--weights")
args = parser.parse_args()

if args.plot_single == 'True':
    args.plot_single = True
else:
    args.plot_single = False

# Set up test configaration
# ----------------------------------------------------------------------------------------------

config = get_config_yaml('config.yaml', {})
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
print("Saving predictions...")
if config['patchify']:
    # predict patch images and merge together 
    
    with open(config['p_test_dir'], 'r') as j:
        patch_test_dir = json.loads(j.read())
    
    df = pd.DataFrame.from_dict(patch_test_dir)
    test_dir = pd.read_csv(config['test_dir'])
    total_score = 0.0
    
    # loop to traverse full dataset
    for i in range(len(test_dir)):
        idx = df[df["masks"]==test_dir["masks"][i]].index
        
        # construct a single full image from prediction patch images
        pred_full_label = np.zeros((512,512), dtype=int)
        for j in idx:
            p_idx = patch_test_dir["patch_idx"][j]
            feature, mask, _ = test_dataset.get_random_data(j)
            pred_mask = model.predict(feature)
            pred_mask = np.argmax(pred_mask, axis = 3)
            pred_full_label[p_idx[0]:p_idx[1], p_idx[2]:p_idx[3]] = pred_mask[0]
        
        
        # read original image and mask
        feature = read_img(test_dir["feature_ids"][i], in_channels=config['in_channels'])
        mask = transform_data(read_img(test_dir["masks"][i], label=True), config['num_classes'])
        
        # calculate keras MeanIOU score
        m = keras.metrics.MeanIoU(num_classes=config['num_classes'])
        m.update_state(np.argmax([mask], axis = 3), [pred_full_label])
        score = m.result().numpy()
        total_score += score
        
        # plot and saving image
        display({"VV": feature[:,:,0],
                    "VH": feature[:,:,1],
                    "DEM": feature[:,:,2],
                    "Mask": np.argmax([mask], axis = 3)[0],
                    "Prediction (MeanIOU_{:.4f})".format(score): pred_full_label
                    }, i, config['prediction_test_dir'], score, config['experiment'])
else:
    show_predictions(test_dataset, model, config)

# metrics = list(get_metrics(config).values())
# adam = keras.optimizers.Adam(learning_rate = config['learning_rate'])
# model.compile(optimizer = adam, loss = focal_loss(), metrics = metrics)
# model.evaluate(test_dataset)



# Visualization code for comparision

# # save models location
# models_paths = ["/home/mdsamiul/github_project/flood_water_mapping_segmentation/model/unet/*.hdf5",
#                 "/home/mdsamiul/github_project/flood_water_mapping_segmentation/model/kuc_u2net/*.hdf5",
#                 "/home/mdsamiul/github_project/flood_water_mapping_segmentation/model/dncnn/*.hdf5",
#                 "/home/mdsamiul/github_project/flood_water_mapping_segmentation/model/kuc_attunet/*.hdf5",
#                 "/home/mdsamiul/github_project/flood_water_mapping_segmentation/model/sm_fpn/*.hdf5",
#                 "/home/mdsamiul/github_project/flood_water_mapping_segmentation/model/sm_linknet/*.hdf5",
#                 "/home/mdsamiul/github_project/flood_water_mapping_segmentation/model/unet++/*.hdf5",
#                 "/home/mdsamiul/github_project/flood_water_mapping_segmentation/model/vnet/*.hdf5",
#                 ]

# # extract models paths 
# models_paths = [x for path in models_paths for x in glob.glob(path)]

# mnet_paths = ["/home/mdsamiul/github_project/flood_water_mapping_segmentation/model/mnet/mnet_ex_regular_epochs_2000_17-Mar-22.hdf5",
#               "/home/mdsamiul/github_project/flood_water_mapping_segmentation/model/mnet/mnet_ex_cls_balance_epochs_2000_17-Mar-22.hdf5",
#               "/home/mdsamiul/github_project/flood_water_mapping_segmentation/model/mnet/mnet_ex_patchify_WOC_256_epochs_2000_12-Apr-22.hdf5",
#               "/home/mdsamiul/github_project/flood_water_mapping_segmentation/model/mnet/mnet_ex_patchify_epochs_2000_02-Apr-22.hdf5"]

# models_paths = models_paths + mnet_paths

# # image index in dataset for compare
# #indices = [14, 35, 45] # best
# indices = [15, 43, 36] # worst

# # initialize dict to save predictions
# plot_tensors = {}
# for idx in indices:
#     plot_tensors[idx] = {"regular":[],
#                 "cls":[],
#                 "patchify":[],
#                 "p_woc":[]}

# # y labels
# cols = ["vv", "vh", "dem", "GR", "unet", "u2net", "dncnn", "attunet","fpn","linknet","unet++","vnet", "mnet"]

# # categories models path based on experiments
# cat_paths = {0:{"regular":[],
#                 "cls":[],},
#             1:{"patchify":[],
#                 "p_woc":[]}}
# for path in models_paths:
#     c1 = path.split("_")[6]
#     c2 = path.split("_")[7]
#     if c1 not in ["regular", "cls", "patchify"]:
#         c1 = path.split("_")[8]
#         c2 = path.split("_")[9]
    
#     if c1 == "patchify" and c2 =="WOC":
#         cat_paths[1]["p_woc"].append(path)
#     elif c1 == "regular":
#         cat_paths[0]["regular"].append(path)
#     elif c1 == "cls":
#         cat_paths[0]["cls"].append(path)
#     else:
#         cat_paths[1]["patchify"].append(path)


# # loading data
# with open(config['p_test_dir'], 'r') as j:
#     patch_test_dir = json.loads(j.read())
    
# df = pd.DataFrame.from_dict(patch_test_dir)
# test_dir = pd.read_csv(config['test_dir'])

# # get dataloader for patchify experiments
# config['patchify'] = True
# test_dataset = get_test_dataloader(config)

# # loop through experiments patchify/patchify_WOC
# for key in cat_paths[1].keys():
    
#     # loop through models path
#     for q, path in enumerate(cat_paths[1][key]):
        
#         # load model
#         model = load_model(path, compile = False)
        
#         # loop through image indices
#         for idx in indices:
#             total_score = 0.0
            
#             # extract all the patch images location
#             Pt_idx = df[df["masks"]==test_dir["masks"][idx]].index
            
#             # declare matrix with original image shape
#             pred_full_label = np.zeros((512,512), dtype=int)
            
#             # loop through patch images for single image and predict
#             for j in Pt_idx:
#                 p_idx = patch_test_dir["patch_idx"][j]
#                 feature, mask, _ = test_dataset.get_random_data(j)
#                 pred_mask = model.predict(feature)
#                 pred_mask = np.argmax(pred_mask, axis = 3)
                
#                 # save prediction into matrix for plotting original image
#                 pred_full_label[p_idx[0]:p_idx[1], p_idx[2]:p_idx[3]] = np.bitwise_or(pred_full_label[p_idx[0]:p_idx[1], p_idx[2]:p_idx[3]], pred_mask[0].astype(int))
            
#             # loading original features and mask   
#             if q==0:
#                 feature = read_img(test_dir["feature_ids"][idx], in_channels=config['in_channels'])
#                 mask = transform_data(read_img(test_dir["masks"][idx], label=True), config['num_classes'])
#                 plot_tensors[idx][key].append(feature[:,:,0])
#                 plot_tensors[idx][key].append(feature[:,:,1])
#                 plot_tensors[idx][key].append(feature[:,:,2])
#                 plot_tensors[idx][key].append(np.argmax([mask], axis = 3)[0])
            
#             # save original prediction after merge patchify predictions
#             plot_tensors[idx][key].append(pred_full_label)

# # get dataloader for regular/cls_balance
# config['patchify'] = False
# test_dataset = get_test_dataloader(config)

# # loop through experiments regular/cls_balance
# for key in cat_paths[0].keys():
    
#     # loop through models path
#     for q, path in enumerate(cat_paths[0][key]):
        
#         # load model
#         model = load_model(path, compile = False)
        
#         # loop through model indices and predict
#         for idx in indices:
#             feature, mask, _ = test_dataset.get_random_data(idx)
#             prediction = model.predict_on_batch(feature)
            
#             # loading original image and indices
#             if q==0:
#                 plot_tensors[idx][key].append(feature[0][:,:,0])
#                 plot_tensors[idx][key].append(feature[0][:,:,1])
#                 plot_tensors[idx][key].append(feature[0][:,:,2])
#                 plot_tensors[idx][key].append(np.argmax(mask, axis = 3)[0])
            
#             # save prediction
#             plot_tensors[idx][key].append(np.argmax(prediction, axis = 3)[0])

# # creating figure for plotting
# fig = plt.figure(figsize=(12, 14))
# fig.subplots_adjust(hspace=0.3, wspace=0)
# gs = fig.add_gridspec(len(cols), len(indices)*4, hspace=0.0, wspace=0.0)
# ax = fig.subplots(nrows=len(cols), ncols=len(indices)*4, sharex='col', sharey='row')

# # flag for tracking column
# i = 0

# # loop through images
# for key in indices:
    
#     # loop through experiments
#     for k in plot_tensors[key].keys():
        
#         # loop through matrix to plot
#         for j in range(len(plot_tensors[key][k])):
            
#             # plot VV and VH
#             if j == 0 or j==1:
#                 ax[j][i].imshow(plot_tensors[key][k][j], cmap='gray', vmin=np.min(plot_tensors[key][k][j]), vmax=np.max(plot_tensors[key][k][j]))
                
#                 # set up title
#                 if j == 0:
#                     ax[j][i].set_title(k, fontsize=10)
                
#                 # remove ticks label and axis from figure
#                 ax[j][i].xaxis.set_major_locator(plt.NullLocator())
#                 ax[j][i].yaxis.set_major_locator(plt.NullLocator())
            
#             # plot DEM
#             elif j == 2:
#                 hillshade = es.hillshade(plot_tensors[key][k][j], altitude=10)
#                 ep.plot_bands(
#                     plot_tensors[key][k][j],
#                     ax=ax[j][i],
#                     cmap="terrain",
#                     cbar=False
#                 )
#                 ax[j][i].imshow(hillshade, cmap="Greys", alpha=0.5)
                
#                 # remove ticks label and axis from figure
#                 ax[j][i].xaxis.set_major_locator(plt.NullLocator())
#                 ax[j][i].yaxis.set_major_locator(plt.NullLocator())
            
#             # plot mask and prediction
#             else:
#                 ax[j][i].imshow(plot_tensors[key][k][j])
                
#                 # remove ticks label and axis from figure
#                 ax[j][i].xaxis.set_major_locator(plt.NullLocator())
#                 ax[j][i].yaxis.set_major_locator(plt.NullLocator())
#         i += 1

# # remove unnecessary space from figure
# plt.subplots_adjust(hspace=0, wspace=0)

# # set up row-wise y label
# for i in range(len(ax)):
#     ax[i][0].set_ylabel(cols[i], fontsize=10, rotation=0)
#     ax[i][0].yaxis.set_label_coords(-0.5, 0.4)

# # save and show plotting figure
# plt.savefig("worst.png", bbox_inches='tight', dpi=1200)
# plt.show()