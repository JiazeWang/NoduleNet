import SimpleITK as sitk
import numpy as np
from save_sitk import savedicom
import os
import scipy.ndimage
from scipy.ndimage.measurements import label
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure

def load_itk_image(filename):
    """Return img array and [z,y,x]-ordered origin and spacing
    """

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing


def resample(image, spacing, new_spacing=[1.0, 1.0, 1.0], order=1):
    """
    Resample image from the original spacing to new_spacing, e.g. 1x1x1
    image: 3D numpy array of raw HU values from CT series in [z, y, x] order.
    spacing: float * 3, raw CT spacing in [z, y, x] order.
    new_spacing: float * 3, new spacing used for resample, typically 1x1x1,
        which means standardizing the raw CT with different spacing all into
        1x1x1 mm.
    order: int, order for resample function scipy.ndimage.interpolation.zoom
    return: 3D binary numpy array with the same shape of the image after,
        resampling. The actual resampling spacing is also returned.
    """
    # shape can only be int, so has to be rounded.
    new_shape = np.round(image.shape * spacing / new_spacing)

    # the actual spacing to resample.
    resample_spacing = spacing * image.shape / new_shape

    resize_factor = new_shape / image.shape

    image_new = scipy.ndimage.interpolation.zoom(image, resize_factor,
                                                 mode='nearest', order=order)

    return (image_new, resample_spacing)

def HU2uint8(image, HU_min=-1200.0, HU_max=600.0, HU_nan=-2000.0):
    """
    Convert HU unit into uint8 values. First bound HU values by predfined min
    and max, and then normalize
    image: 3D numpy array of raw HU values from CT series in [z, y, x] order.
    HU_min: float, min HU value.
    HU_max: float, max HU value.
    HU_nan: float, value for nan in the raw CT image.
    """
    image_new = np.array(image)
    image_new[np.isnan(image_new)] = HU_nan

    # normalize to [0, 1]
    image_new = (image_new - HU_min) / (HU_max - HU_min)
    image_new = np.clip(image_new, 0, 1)
    image_new = (image_new * 255).astype('uint8')

    return image_new


def main():
    filelist = "/research/dept8/jzwang/code/lung_nodule_detector/dlung_v1/data/patient_mhd.txt"
    with open(filelist, "r") as f:
        lines = f.readlines()
    params_lists = []
    for line in lines:
        line = line.rstrip()
        filefullname = "/research/dept8/jzwang/code/lung_nodule_detector/dlung_v1/data/mhd/"+line
        input, orgin, spacing= load_itk_image(filefullname)
        input = HU2uint8(HU2uint8)
        resamplenew, resampled_spacing = resample(input, spacing, order=3)
        outputdir="result_resample/"+line[0:-4]
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        orgin = np.array(list(reversed(orgin)))
        resampled_spacing = np.array(list(reversed(resampled_spacing)))
        print("resampled_spacing:", resampled_spacing)
        savedicom(outputdir, resamplenew, resampled_spacing, orgin, pixel_dtypes="int16")

if __name__=='__main__':
    main()
