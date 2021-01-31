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
        pro = 1
        spacing = os.path.join("/research/dept8/jzwang/dataset/LUNA16/preprocessed_test", '%s_spacing_origin.npy' % (result[i][0]))
        origin = os.path.join("/research/dept8/jzwang/dataset/LUNA16/preprocessed_test", '%s_origin.npy' % (result[i][0]))
        spacing = np.array(list(reversed(spacing)))
        origin = np.array(list(reversed(origin)))
        print("xyz.shape", xyz.shape)
        print("spacing.shape", spacing.shape)
        print("origin.shape", origin.shape)
        xyz = xyz*spacing+origin
        new.append([result[i][0], xyz[0], xyz[1], xyz[2], size, pro])
    #new = np.concatenate(new, axis=0)
    col_names = ['seriesuid','coordX','coordY','coordZ','diameter_mm', 'probability']
    submission_path = outputname
    df = pd.DataFrame(new, columns=col_names)
    df.to_csv(submission_path, index=False)

convert_csv_2_origin("3_annotation.csv", "3_annotation_after_convert.csv")
