import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from pathlib import Path


def run_kmeans(image_loc, img_num, save_path):
    im = cv2.imread(image_loc)
    mask_image = im[:, :, 0]
    mask_flat = mask_image.flatten()
    mask_re = mask_flat.reshape(-1, 1)
    arr = mask_re
    index = 0
    for i in mask_re:
        if i < 255:
            arr[index] = 255 - i
        else:
            arr[index] = 0
        index += 1

    # using k means clustering to split the image into 2 clusters --> cloud and sky
    k_means = KMeans(init='k-means++', n_clusters=2)  # can use 8 here for all colour channels
    k_means.fit(arr)
    k_means_labels = k_means.labels_
    k_means_values = k_means.cluster_centers_.squeeze()
    # transforming values to pure white and black (to 0 and 255)
    k_means_values[0] = k_means_values[0]/k_means_values[0] - 1
    k_means_values[1] = (k_means_values[1]/k_means_values[1]) * 255
    img_seg = np.choose(k_means_labels, k_means_values)  # .astype('uint8')  # include if using 8 clusters
    img_seg.shape = mask_image.shape

    #vmin = im.min()
    #vmax = im.max()
    
    img = Image.fromarray(img_seg.astype('uint8'))
    img = img.convert("L")
    Path(save_path).mkdir(parents=True, exist_ok=True)
    img.save(save_path + str(img_num) + '.png')
    '''
    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img_seg, cmap="Greys", vmin=vmin, vmax=vmax)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
    fig.savefig("binary_maps/%d_binary_map.png" % img_num, bbox_inches='tight')
    '''
