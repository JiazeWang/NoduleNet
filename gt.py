import os
import shutil
import numpy as np

def preprocess(params):
    pid = params
    base = "/research/dept8/jzwang/dataset/LUNA16/preprocessed_test_v1"
    origin = np.load(os.path.join(base, "%s_origin.npy"%pid))
    spacing = np.load(os.path.join(base, "%s_spacing_origin.npy"%pid))
    this_annos = np.copy(abbrevs[abbrevs[:, 0] == str(pid)])
    luna_annos = np.copy(this_annos)
    annos_shape = np.shape(luna_annos)
    label = []
    for i in range(len(luna_annos)):
        luna_annos[i][1] = (luna_annos[i][1] - origin[2]) / spacing[2]
        luna_annos[i][2] = (luna_annos[i][2] - origin[1]) / spacing[1]
        luna_annos[i][3] = (luna_annos[i][3] - origin[0]) / spacing[0]
        label.append(np.concatenate()







with open("/research/dept8/jzwang/code/NoduleNet/datav1.txt", "r") as f:
    lines = f.readlines()
for pid in lines:
    pid = pid.rstrip()
    params_lists.append([pid[0:-4]])

pool = Pool(processes=10)
pool.map(preprocess, params_lists)

pool.close()
pool.join()
