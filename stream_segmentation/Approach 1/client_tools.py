import numpy as np
import SimpleITK as sitk

def crop_ultrasound_stream(img):
    top_boundary = int(img.shape[0] * 0.1)
    bottom_boundary = int(img.shape[0] * 0.7)
    left_boundary = int(img.shape[1] * 0.2)
    right_boundary = int(img.shape[1] * 0.8)
    return img[top_boundary:bottom_boundary, left_boundary:right_boundary]

def crop_image(img):
    to_crop = int(img.shape[0] * 0.1) # removes this many rows from the top
    return img[to_crop:, :]
