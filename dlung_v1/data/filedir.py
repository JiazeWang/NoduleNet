import os

def sitk_dcm_dir(dcm_dir):
    path = []
    filename=[]
    for i in os.listdir(dcm_dir):
        dicom_count = 0
        for j in os.listdir(os.path.join(dcm_dir, i)):
            for k in os.listdir(os.path.join(dcm_dir,i,j)):
                num = len(os.listdir(os.path.join(dcm_dir,i,j,k)))
                if num > dicom_count:
                    dicom_count = num
                    dicom_path = os.path.join(dcm_dir,i,j,k)
                    savename = ".".join([dcm_dir,i,j,k])
        path.append(dicom_path)
        filename.append(savename)
    return sorted(path), filename

if __name__=="__main__":
    path, filename=sitk_dcm_dir("sample")
    print(filename)
    with open("filedir.txt",'w') as f:
        f.write('\n'.join(path))
    with open("eval.csv",'w') as f:
        f.write('\n'.join(filename))


