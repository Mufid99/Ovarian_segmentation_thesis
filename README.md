# Ovarian Segmentation Thesis
This repository contains code for the preparation of the Ovarian Cyst dataset to be used in the production of a model which could segment Ovarian cyst features
with a high degree of accuracy. The dataset is not available publicly and therefore this repository will not contain the dataset utilised or the model produced.
The model is generated by using **[nnU-Net](https://github.com/MIC-DKFZ/nnUNet)**. This repository also has code to extend the 
functionality of nnU-Net allowing the segmentation of single tensors without the need to read and write to and from image files. This allows the creation of a tool that
can segment a video stream being sent from an ultrasound machine which is also implemented in this repository. 
