from loss import *
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import rasterio
import matplotlib.pyplot as plt
import earthpy.plot as ep
import earthpy.spatial as es


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

def zero_pad(img_dir, height, width, in_channels=None, label=None):
    if label:
        img = np.zeros((height, width), dtype=int)
        if isinstance(img_dir, str):
            with rasterio.open(img_dir) as fmask:
                mask = fmask.read(1)
                mask[mask == 255] = 0
        else:
            mask = img_dir
        h = (height-mask.shape[0])//2
        w = (width-mask.shape[1])//2
        img[h:h+mask.shape[0], w:w+mask.shape[1]] = mask
        return img
    else:
        img = np.zeros((height,width, in_channels))
        
        # read N number of channels
        for i in range(in_channels):
            tmp_ext = label_norm[i][0]
            with rasterio.open((img_dir+tmp_ext)) as f:
                fea = f.read(1)

            if fea.shape[0]>height:
                fea = fea[:height,:]
            if fea.shape[1]>width:
                fea = fea[:, :width]
            fea = (fea - label_norm[i][1]) / label_norm[i][2]
            h = (height-fea.shape[0])//2
            w = (width-fea.shape[1])//2
            # normalize data
            img[h:h+fea.shape[0], w:w+fea.shape[1],i] = fea

        return img
sample_img_path = ["/home/mdsamiul/github_project/flood_water_mapping_segmentation/data/case_study/sample1/201709_before_flood","/home/mdsamiul/github_project/flood_water_mapping_segmentation/data/case_study/sample2/201709_after_flood"]

models_paths = ["/home/mdsamiul/github_project/flood_water_mapping_segmentation/model/unet/unet_ex_patchify_epochs_2000_15-Mar-22.hdf5",
                "/home/mdsamiul/github_project/flood_water_mapping_segmentation/model/kuc_u2net/kuc_u2net_ex_patchify_epochs_2000_18-Mar-22.hdf5",
                "/home/mdsamiul/github_project/flood_water_mapping_segmentation/model/dncnn/dncnn_ex_patchify_epochs_2000_16-Mar-22.hdf5",
                "/home/mdsamiul/github_project/flood_water_mapping_segmentation/model/kuc_attunet/kuc_attunet_ex_patchify_epochs_2000_24-Mar-22.hdf5",
                "/home/mdsamiul/github_project/flood_water_mapping_segmentation/model/sm_fpn/sm_fpn_ex_patchify_epochs_2000_21-Mar-22.hdf5",
                "/home/mdsamiul/github_project/flood_water_mapping_segmentation/model/sm_linknet/sm_linknet_ex_patchify_epochs_2000_27-Mar-22.hdf5",
                "/home/mdsamiul/github_project/flood_water_mapping_segmentation/model/unet++/unet++_ex_patchify_epochs_2000_19-Mar-22.hdf5",
                "/home/mdsamiul/github_project/flood_water_mapping_segmentation/model/vnet/vnet_ex_patchify_epochs_2000_15-Mar-22.hdf5",
                ]
mnet_paths = ["/home/mdsamiul/github_project/flood_water_mapping_segmentation/model/mnet/mnet_ex_patchify_epochs_2000_02-Apr-22.hdf5"]
models_paths = models_paths + mnet_paths
plot_tensors = {"201709_before_flood":[],
               "201709_after_flood":[],}
cols = ["vv", "vh", "dem", "unet", "u2net", "dncnn", "attunet","fpn","linknet","unet++","vnet", "mnet"]

for q, path in enumerate(models_paths):
    model = load_model(path, compile = False)
    for p in sample_img_path:
        id = p.split("/")[-1]
        feature = zero_pad(p, height=256, width=256, in_channels=3)
        if q==0:
            # with rasterio.open((p+"_vv.tif")) as vv:
            #     vv_img = vv.read(1)
            #     plot_tensors[id].append(vv_img)
            # with rasterio.open((p+"_vh.tif")) as vh:
            #     vh_img = vh.read(1)
            #     plot_tensors[id].append(vv_img)
            # with rasterio.open((p+"_nasadem.tif")) as dem:
            #     dem_img = dem.read(1)
            #    plot_tensors[id].append(vv_img)
            plot_tensors[id].append(feature[:,:,0])
            plot_tensors[id].append(feature[:,:,1])
            plot_tensors[id].append(feature[:,:,2])
        feature = zero_pad(p, height=256, width=256, in_channels=3)
        pred = model.predict(tf.expand_dims(feature, axis=0))
        plot_tensors[id].append(np.argmax(pred, axis = 3)[0])

plot_tensors["201709_after_flood"][2] = plot_tensors["201709_before_flood"][2]
# for key in plot_tensors.keys():
#     shape = plot_tensors[key][0].shape
#     for i in range(3,len(plot_tensors[key])):
#         plot_tensors[key][i] = zero_pad(unpad(plot_tensors[key][i], (shape)), shape[0], shape[1], label=True)

# for key in plot_tensors.keys():
#     for i in range(len(plot_tensors[key])):
#         print(plot_tensors[key][i].shape)

# creating figure for plotting
fig = plt.figure(figsize=(3, 14))
fig.subplots_adjust(hspace=0.3, wspace=0)
gs = fig.add_gridspec(len(cols), 2, hspace=0.0, wspace=0.0)
ax = fig.subplots(nrows=len(cols), ncols=2, sharex='col', sharey='row')

# flag for tracking column
i = 0

# loop through images
for path in sample_img_path:
    key = path.split("/")[-1]
    
    # loop through experiments
    for j, k in enumerate(plot_tensors[key]):
        if j == 0 or j==1:
            ax[j][i].imshow(k, cmap='gray', vmin=np.min(k), vmax=np.max(k))
                
            # set up title
            if j == 0:
                title = ""
                if key.split("_")[-2] == "before":
                    title = "Before Flood"
                else:
                    title = "After Flood"
                ax[j][i].set_title(title, fontsize=10)
                
            # remove ticks label and axis from figure
            ax[j][i].xaxis.set_major_locator(plt.NullLocator())
            ax[j][i].yaxis.set_major_locator(plt.NullLocator())
        
        # plot DEM
        elif j == 2:
            hillshade = es.hillshade(k, altitude=10)
            ep.plot_bands(
                k,
                ax=ax[j][i],
                cmap="terrain",
                cbar=False
            )
            ax[j][i].imshow(hillshade, cmap="Greys", alpha=0.5)
                
            # remove ticks label and axis from figure
            ax[j][i].xaxis.set_major_locator(plt.NullLocator())
            ax[j][i].yaxis.set_major_locator(plt.NullLocator())
            
        # plot mask and prediction
        else:
            ax[j][i].imshow(k)
                
            # remove ticks label and axis from figure
            ax[j][i].xaxis.set_major_locator(plt.NullLocator())
            ax[j][i].yaxis.set_major_locator(plt.NullLocator())
    i += 1

# remove unnecessary space from figure
plt.subplots_adjust(hspace=0, wspace=0)

# set up row-wise y label
for i in range(len(ax)):
    ax[i][0].set_ylabel(cols[i], fontsize=10, rotation=0)
    ax[i][0].yaxis.set_label_coords(-0.5, 0.4)

# save and show plotting figure
plt.savefig("case.png", bbox_inches='tight', dpi=1200)
plt.show()                