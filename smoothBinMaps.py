import numpy as np
from PIL import Image
import os
from pathlib import Path

def smoothImage(image):
    imgCopy = image.copy()
    windowSize = 31 # Should be an odd number
    #for i in range((windowSize-1)/2, imgCopy.shape[0]-(windowSize-1)/2):
    #    for j in range((windowSize-1)/2, imgCopy.shape[1]-(windowSize-1)/2):
    for i in range(imgCopy.shape[0]):
        for j in range(imgCopy.shape[1]):
            rowStartIdx = int(np.max([0, i-(windowSize-1)/2]))
            rowEndIdx = int(np.min([imgCopy.shape[0]-1, i+(windowSize-1)/2]))
            colStartIdx = int(np.max([0, j-(windowSize-1)/2]))
            colEndIdx = int(np.min([imgCopy.shape[1]-1, j+(windowSize-1)/2]))
            temp = image[rowStartIdx:rowEndIdx, colStartIdx:colEndIdx].flatten()
            imgCopy[i, j] = int(np.bincount(temp).argmax())
    imgCopy = imgCopy.astype('uint8')
    return imgCopy

if __name__ == '__main__':
    binMaps_original = './data/BINmaps_generated_images/'
    binMaps_smoothened = './data/genBINmaps_smoothened_images/'
    
    Path(binMaps_smoothened).mkdir(parents=True, exist_ok=True)
    file_list = [f for f in os.listdir(binMaps_original) if os.path.isfile(os.path.join(binMaps_original, f))]
    for file in file_list:
        # Read and Store Target Image
        image_obj = Image.open(binMaps_original + file)
        image = np.asarray(image_obj)
        smooth_binMap = smoothImage(image)
        im = Image.fromarray(smooth_binMap)
        im = im.convert("L")
        im.save(binMaps_smoothened + file)
        print("Smoothening Successful on Image " + file)
    '''
    ori_binMaps = transformations.data_load(binMaps_original)
    
    Path(binMaps_smoothened).mkdir(parents=True, exist_ok=True)
    for (idx,image) in enumerate(ori_binMaps):
        smooth_binMap = smoothImage(image)
        im = Image.fromarray(smooth_binMap)
        im = im.convert("L")
        im.save(binMaps_smoothened + str(idx+1) + '.png')
        print("Smoothening Successful on Image #%d!" % (idx+1))
    '''