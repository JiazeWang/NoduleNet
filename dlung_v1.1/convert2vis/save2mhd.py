import SimpleITK as sitk
import numpy as np
from save_sitk import savedicom
import os

def load_itk_image(filename):
    """Return img array and [z,y,x]-ordered origin and spacing
    """

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing



def main():
    filelist = "/research/dept8/jzwang/code/lung_nodule_detector/dlung_v1/data/patient_mhd.txt"
    with open(filelist, "r") as f:
        lines = f.readlines()
    params_lists = []
    for line in lines:
        line = line.rstrip()
        filefullname = "/research/dept8/jzwang/code/lung_nodule_detector/dlung_v1/data/mhd/"+line
        input, orgin, spacing= load_itk_image(filefullname)
        outputdir="result/"+line[0:-4]
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        savedicom(outputdir, input, spacing, orgin, pixel_dtypes="int16")

if __name__=='__main__':
    main()
