import pandas as pd
import numpy as np
import os

def convert_csv_2_origin(filename, outputname):
    CSV_FILE_PATH = filename
    df = pd.read_csv(CSV_FILE_PATH)
    result = df.values.tolist()
    new = []
    for i in range(0, len(result)):
        xyz = np.array([result[i][1], result[i][2], result[i][3]])
        size = result[i][4]
        spacing = os.path.join("/research/dept8/jzwang/dataset/LUNA16/preprocessed_test", '%s_spacing_origin.npy' % (result[i][0]))
        origin = os.path.join("/research/dept8/jzwang/dataset/LUNA16/preprocessed_test", '%s_origin.npy' % (result[i][0]))
        if os.path.exists(spacing):
            spacing = np.load(spacing)
            origin = np.load(origin)
            spacing = np.array(list(reversed(spacing)))
            origin = np.array(list(reversed(origin)))
            xyz = (xyz-origin)/spacing
            new.append([result[i][0], xyz[0], xyz[1], xyz[2], size])
    #new = np.concatenate(new, axis=0)
    col_names = ['seriesuid','coordX','coordY','coordZ','diameter_mm']
    submission_path = outputname
    df = pd.DataFrame(new, columns=col_names)
    df.to_csv(submission_path, index=False)

convert_csv_2_origin("annotations_origin.csv", "annotation2ours.csv")
