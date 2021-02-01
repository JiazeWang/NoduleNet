import numpy as np
import torch
import os
import traceback
import time
import nrrd
import sys
import matplotlib.pyplot as plt
import logging
import argparse
import torch.nn.functional as F
import SimpleITK as sitk
from scipy.stats import norm
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parallel.data_parallel import data_parallel
from scipy.ndimage.measurements import label
from scipy.ndimage import center_of_mass
from net.nodule_net import NoduleNet
from dataset.collate import train_collate, test_collate, eval_collate
from dataset.bbox_reader import BboxReader
from dataset.mask_reader import MaskReader
from config_lidc import config
from utils.util import onehot2multi_mask, normalize, pad2factor, load_dicom_image, crop_boxes2mask_single, npy2submission
import pandas as pd
import json
from evaluationScript.noduleCADEvaluationLUNA16 import noduleCADEvaluation

plt.rcParams['figure.figsize'] = (24, 16)
plt.switch_backend('agg')
this_module = sys.modules[__name__]
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


parser = argparse.ArgumentParser()
parser.add_argument('--net', '-m', metavar='NET', default=config['net'],
                    help='neural net')
parser.add_argument("mode", type=str,
                    help="you want to test or val")
parser.add_argument("--weight", type=str, default=config['initial_checkpoint'],
                    help="path to model weights to be used")
parser.add_argument("--dicom-path", type=str, default=None,
                    help="path to dicom files of patient")
parser.add_argument("--out-dir", type=str, default=config['out_dir'],
                    help="path to save the results")
parser.add_argument("--test-set-name", type=str, default=config['test_set_name'],
                    help="path to save the results")


def main():
    logging.basicConfig(format='[%(levelname)s][%(asctime)s] %(message)s', level=logging.INFO)
    args = parser.parse_args()
    if args.mode == 'eval':
        data_dir = config['preprocessed_data_dir']
        test_set_name = args.test_set_name
        num_workers = 0
        initial_checkpoint = args.weight
        net = args.net
        out_dir = args.out_dir
        net = getattr(this_module, net)(config)
        net = net.cuda()
        if initial_checkpoint:
            print('[Loading model from %s]' % initial_checkpoint)
            checkpoint = torch.load(initial_checkpoint)
            epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
        else:
            print('No model weight file specified')
            return
        print('out_dir', out_dir)
        save_dir = os.path.join(out_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(os.path.join(save_dir)):
            os.makedirs(os.path.join(save_dir))
        dataset = MaskReader(data_dir, test_set_name, config, mode='eval')
        eval(net, dataset, save_dir)
    else:
        logging.error('Mode %s is not supported' % (args.mode))

def convert_json(input, output, thresholds=0.5):
    with open(input, "r") as f:
        lines = f.readlines()
    NoduleClass, NoduleScore, NoduleCoordinates, NoduleDiameter= [], [], [], []
    nudule = {}
    num = 0
    result = []
    record = lines[1].split(",")[0]
    for line in lines[1:]:
        nodule_dic = {}
        line = line.rstrip()
        line = line.split(",")
        if line[0] == record and num+1<len(lines[1:]) :
            NoduleScore.append(line[-1])
            if float(line[-1])> thresholds:
                NoduleClass.append(1)
            else:
                NoduleClass.append(0)
            NoduleCoordinates.append([line[1], line[2], line[3]])
            NoduleDiameter.append(line[4])
        else:
            nudule = {}
            patient = {"patientName": record, \
                       "nodules": nudule,}
            nudule["NoduleScore"] = NoduleScore
            nudule["NoduleClass"] = NoduleClass
            nudule["NoduleCoordinates"] = NoduleCoordinates
            nudule["NoduleDiameter"] = NoduleDiameter
            NoduleClass, NoduleScore, NoduleCoordinates, NoduleDiameter = [], [], [], []
            NoduleScore.append(line[-1])
            NoduleCoordinates.append([line[1], line[2], line[3]])
            NoduleDiameter.append(line[4])
            record = line[0]
            result.append(patient)
        num = num + 1
    with open(output,'w',encoding='utf-8') as f:
        f.write(json.dumps(result,indent=2))

def convert_csv_2_origin(filename, outputname):
    CSV_FILE_PATH = filename
    df = pd.read_csv(CSV_FILE_PATH)
    result = df.values.tolist()
    new = []
    resolution = np.array([1,1,1])
    for i in range(0, len(result)):
        xyz = np.array([result[i][1], result[i][2], result[i][3]])
        size = result[i][4]
        pro = result[i][5]
        xyz = xyz
        spacing = np.load(os.path.join(config['preprocessed_data_dir'], '%s_spacing_origin.npy' % (result[i][0])))
        origin = np.load(os.path.join(config['preprocessed_data_dir'], '%s_origin.npy' % (result[i][0])))
        spacing = np.array(list(reversed(spacing)))
        origin = np.array(list(reversed(origin)))
        xyz = xyz * resolution / spacing
        new.append([result[i][0], xyz[0], xyz[1], xyz[2], size, pro])
    #new = np.concatenate(new, axis=0)
    col_names = ['seriesuid','coordX','coordY','coordZ','diameter_mm', 'probability']
    submission_path = outputname
    df = pd.DataFrame(new, columns=col_names)
    df.to_csv(submission_path, index=False)

def eval(net, dataset, save_dir=None):
    net.set_mode('eval')
    net.use_mask = False
    net.use_rcnn = True
    aps = []
    dices = []
    raw_dir = config['data_dir']
    preprocessed_dir = config['preprocessed_data_dir']

    print('Total # of eval data %d' % (len(dataset)))
    for i, (input, image) in enumerate(dataset):
        try:
            D, H, W = image.shape
            pid = dataset.filenames[i]
            print('[%d] Predicting %s' % (i, pid), image.shape)

            with torch.no_grad():
                input = input.cuda().unsqueeze(0)
                net.forward(input)

            rpns = net.rpn_proposals.cpu().numpy()
            detections = net.detections.cpu().numpy()
            ensembles = net.ensemble_proposals.cpu().numpy()

            if len(detections) and net.use_mask:
                crop_boxes = net.crop_boxes
                segments = [F.sigmoid(m).cpu().numpy() > 0.5 for m in net.mask_probs]
                pred_mask = crop_boxes2mask_single(crop_boxes[:, 1:], segments, input.shape[2:])
                pred_mask = pred_mask.astype(np.uint8)
            else:
                pred_mask = np.zeros((input[0].shape))
            print('rpn', rpns.shape)
            print('detection', detections.shape)
            print('ensemble', ensembles.shape)


            if len(rpns):
                rpns = rpns[:, 1:]
                np.save(os.path.join(save_dir, '%s_rpns.npy' % (pid)), rpns)

            if len(detections):
                detections = detections[:, 1:-1]
                np.save(os.path.join(save_dir, '%s_rcnns.npy' % (pid)), detections)

            if len(ensembles):
                ensembles = ensembles[:, 1:]
                np.save(os.path.join(save_dir, '%s_ensembles.npy' % (pid)), ensembles)


            # Clear gpu memory
            del input#, gt_mask, gt_img, pred_img, full, score
            torch.cuda.empty_cache()

        except Exception as e:
            del input, image,
            torch.cuda.empty_cache()
            traceback.print_exc()

            print
            return

    rpn_res = []
    rcnn_res = []
    ensemble_res = []
    for pid in dataset.filenames:
        if os.path.exists(os.path.join(save_dir, '%s_rpns.npy' % (pid))):
            rpns = np.load(os.path.join(save_dir, '%s_rpns.npy' % (pid)))
            rpns = rpns[:, [3, 2, 1, 4, 0]]
            names = np.array([[pid]] * len(rpns))
            rpn_res.append(np.concatenate([names, rpns], axis=1))

        if os.path.exists(os.path.join(save_dir, '%s_rcnns.npy' % (pid))):
            rcnns = np.load(os.path.join(save_dir, '%s_rcnns.npy' % (pid)))
            rcnns = rcnns[:, [3, 2, 1, 4, 0]]
            names = np.array([[pid]] * len(rcnns))
            rcnn_res.append(np.concatenate([names, rcnns], axis=1))

        if os.path.exists(os.path.join(save_dir, '%s_ensembles.npy' % (pid))):
            ensembles = np.load(os.path.join(save_dir, '%s_ensembles.npy' % (pid)))
            ensembles = ensembles[:, [3, 2, 1, 4, 0]]
            names = np.array([[pid]] * len(ensembles))
            ensemble_res.append(np.concatenate([names, ensembles], axis=1))

    rpn_res = np.concatenate(rpn_res, axis=0)
    rcnn_res = np.concatenate(rcnn_res, axis=0)
    ensemble_res = np.concatenate(ensemble_res, axis=0)
    col_names = ['seriesuid','coordX','coordY','coordZ','diameter_mm', 'probability']
    eval_dir = config['data_dir']
    ensemble_submission_path = os.path.join(eval_dir, 'submission.csv')
    df = pd.DataFrame(ensemble_res, columns=col_names)
    df.to_csv(ensemble_submission_path, index=False)
    ensemble_submission_path_v1 = os.path.join(eval_dir, 'submission_v1.csv')
    convert_csv_2_origin(ensemble_submission_path, ensemble_submission_path_v1)
    convert_json(ensemble_submission_path_v1, "result.json")

    print

if __name__ == '__main__':
    main()
