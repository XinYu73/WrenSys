import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import csv
import gc
from tqdm import tqdm
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from wren_sys.data import collate_batch
from wren_sys.data import WyckoffData
from wren_sys.model import ClassificationModel
import pandas as pd

cuda = torch.cuda.is_available()


# load data to momery
def preload(preload_folder, id_prop_file):
    data = []
    datasetcsv = pd.read_csv(id_prop_file)
    cif_list = datasetcsv["mpID"]
    for cif_id in tqdm(cif_list):
        with open(preload_folder + "/" + cif_id + ".pickle", "rb") as f:
            data.append(pickle.load(f))
    return data


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def predict_one_bag(val_loader, model, bagging, return_pd=False):
    test_targets = []
    test_preds = []
    test_cif_ids = []
    # switch to evaluate mode
    model.eval()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        with torch.no_grad():
            if cuda:
                input_var = (item.cuda(non_blocking=True) for item in input)
            else:
                input_var = input
            target_normed = target.view(-1).long()
            if cuda:
                target_var = target_normed.cuda(non_blocking=True)
            else:
                target_var = target_normed
        output = model(*input_var)
        # measure accuracy and record loss
        test_pred = torch.exp(output.data.cpu())
        test_target = target
        assert test_pred.shape[1] == 2
        test_preds += test_pred[:, 1].tolist()
        test_targets += test_target.view(-1).tolist()
        test_cif_ids += batch_cif_ids

    # if not return_pd:
    #     import csv

    #     with open("test_results_bag_" + str(bagging) + ".csv", "w") as f:
    #         writer = csv.writer(f)
    #         for cif_id, target, pred in zip(test_cif_ids, test_targets, test_preds):
    #             writer.writerow((cif_id, target, pred))
    return pd.DataFrame(
        {
            "test_cif_ids": test_cif_ids,
            "test_targets": test_targets,
            "test_preds": test_preds,
        }
    )


def bootstrap_aggregating(bagging_size, prediction=False):
    predval_dict = {}

    # print("Do bootstrap aggregating for %d models.............." % (bagging_size))
    for i in range(1, bagging_size + 1):
        if prediction:
            filename = "test_results_prediction_" + str(i) + ".csv"
        else:
            filename = "test_results_bag_" + str(i) + ".csv"
        df = pd.read_csv(os.path.join(filename), header=None)
        id_list = df.iloc[:, 0].tolist()
        pred_list = df.iloc[:, 2].tolist()
        for idx, mat_id in enumerate(id_list):
            if mat_id in predval_dict:
                predval_dict[mat_id].append(float(pred_list[idx]))
            else:
                predval_dict[mat_id] = [float(pred_list[idx])]

    # print("Writing CLscore file....")
    with open("test_results_ensemble_" + str(bagging_size) + "models.csv", "w") as g:
        g.write("id,CLscore,bagging")  # mp-id, CLscore, # of bagging size

        for key, values in predval_dict.items():
            g.write("\n")
            g.write(key + "," + str(np.mean(np.array(values))) + "," + str(len(values)))
    # print("Done")


def bootstrap_aggregating_pds(pds):
    predval_dict = {}
    for df in pds:
        id_list = df.iloc[:, 0].tolist()
        pred_list = df.iloc[:, 2].tolist()
        for idx, mat_id in enumerate(id_list):
            if mat_id in predval_dict:
                predval_dict[mat_id].append(float(pred_list[idx]))
            else:
                predval_dict[mat_id] = [float(pred_list[idx])]

    ids = []
    clscores = []
    for key, values in predval_dict.items():
        ids.append(key)
        clscores.append(np.mean(np.array(values)))

    return pd.DataFrame({"ID": ids, "CLscore": clscores})


def predict_model(
    csvpath,
    model_dir=None,
    start_bag=0,
    bag=2,
    batch_size=256,
    workers=10,
    pin_memory_flag=True,
    prefetch_factor=10,
    return_pd=False,
):
    results_pd = []
    for i in tqdm(range(start_bag + 1, start_bag + bag + 1)):
        collate_fn = collate_batch
        dataset_test = WyckoffData(pd.read_csv(csvpath))
        # dataset_test = preload(preload_folder=graph_dir, id_prop_file=csvpath)
        test_loader = DataLoader(
            dataset_test,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory_flag,
            prefetch_factor=prefetch_factor,
        )
        # build model
        model = ClassificationModel()
        if cuda:
            model.cuda()
        modelpath = os.path.join(
            model_dir, "model_highest_AUC_bag_" + str(i) + ".pth.tar"
        )
        if os.path.isfile(modelpath):
            checkpoint = torch.load(
                modelpath, map_location=lambda storage, loc: storage
            )
            model.load_state_dict(checkpoint["state_dict"])
        else:
            print("=> no model found at '{}'".format(modelpath))
        results_pd.append(predict_one_bag(test_loader, model, i, return_pd))
    # if return_pd:
    #     return results_pd
    # bootstrap_aggregating(bag, prediction=True)
    return bootstrap_aggregating_pds(results_pd)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", help="model path", type=str)
    parser.add_argument("-p", "--in_csv_path", help="csv path", type=str)
    parser.add_argument("-n", "--out_csv_path", help="out out csv path", type=str)
    args = parser.parse_args()
    pdout = predict_model(
        csvpath=args.in_csv_path,
        model_dir=args.model_path,
        start_bag=0,
        bag=56,
        batch_size=128,
        workers=2,
        pin_memory_flag=True,
        prefetch_factor=2,
        return_pd=False,
    )
    ini = pd.read_csv(args.in_csv_path)
    ini.sort_values("mpID", inplace=True)
    pdout.sort_values("ID", inplace=True)
    ini.insert(1, "CLscore", pdout.iloc[:, 1].to_list())
    ini.sort_values("CLscore", inplace=True, ascending=False)
    ini.to_csv(args.out_csv_path, index=False)
