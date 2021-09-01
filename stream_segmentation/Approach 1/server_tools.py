import numpy as np
import SimpleITK as sitk
import os
import re
import torch

# obtained from installed nnU-Net library but added changes to fit dataset
# used to create the nifti files for both segmentations and images to be used by nnunet
def convert_2d_image_to_nifti_new(img, output_filename_truncated: str, spacing=(999, 1, 1),
                              transform=None, is_seg: bool = False) -> None:
    """
    Reads an image (must be a format that it recognized by skimage.io.imread) and converts it into a series of niftis.
    The image can have an arbitrary number of input channels which will be exported separately (_0000.nii.gz,
    _0001.nii.gz, etc for images and only .nii.gz for seg).
    Spacing can be ignored most of the time.
    !!!2D images are often natural images which do not have a voxel spacing that could be used for resampling. These images
    must be resampled by you prior to converting them to nifti!!!
    Datasets converted with this utility can only be used with the 2d U-Net configuration of nnU-Net
    If Transform is not None it will be applied to the image after loading.
    Segmentations will be converted to np.uint32!
    :param is_seg:
    :param transform:
    :param input_filename:
    :param output_filename_truncated: do not use a file ending for this one! Example: output_name='./converted/image1'. This
    function will add the suffix (_0000) and file ending (.nii.gz) for you.
    :param spacing:
    :return:
    """
    # img = io.imread(input_filename)

    if transform is not None:
        img = transform(img)

    # if input_filename.endswith(".nrrd"):
    #     img = np.squeeze(img)
    elif img.shape[2] == 4:
        img = img[:,:,0:3]

    if len(img.shape) == 2:  # 2d image with no color channels
        img = img[None, None]  # add dimensions
    else:
        assert len(img.shape) == 3, "image should be 3d with color channel last but has shape %s" % str(img.shape)
        # we assume that the color channel is the last dimension. Transpose it to be in first
        img = img.transpose((2, 0, 1))
        # add third dimension
        img = img[:, None]
    # image is now (c, x, x, z) where x=1 since it's 2d
    if is_seg:
        assert img.shape[0] == 1, 'segmentations can only have one color channel, not sure what happened here'

    for j, i in enumerate(img):

        if is_seg:
            i = i.astype(np.uint32)

        itk_img = sitk.GetImageFromArray(i)
        itk_img.SetSpacing(list(spacing)[::-1])
        if not is_seg:
            sitk.WriteImage(itk_img, output_filename_truncated + "_%04.0d.nii.gz" % j)
        else:
            sitk.WriteImage(itk_img, output_filename_truncated + ".nii.gz")


# obtained from https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def get_int_version(text):
    return int(text) if text.isdigit() else text

# obtained from https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def human_keys(text):
    return [ get_int_version(x) for x in re.split(r'(\d+)', text) ]
