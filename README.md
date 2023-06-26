# Using GANs to Augment Data for Cloud Image Segmentation Task

With the spirit of reproducible research, this repository contains all the codes required to produce the results in the manuscript:

> Jain, M., Meegan, C. and Dev, S.(2021). Using GANs to Augment Data for Cloud Image Segmentation Task. In: IEEE International Geoscience and Remote Sensing Symposium (IGARSS), 2021

The description of each of the code files is as follows:

- `GAN.py`: Reads the sky/cloud images, trains a GAN and then use it to generate the new images
- `clustering.py`: Contains the code to perform sky/cloud image segmentation using k-Means clustering (unsupervised)
- `transformations.py`: Contains some utility functions to perform basic image transformations for data augmentation
- `smoothBinMaps.py`: Smoothens the segmentation maps that were estimated using the `clustering.py`
- `main.py`: Trains and evaluate PLS regression method with and without GAN augmentation. It further checks if the generated sky/cloud & GT map pair falls within the distribution of the original dataset.
