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


def save_checkpoint(state, is_best, bagging, epoch):
    if is_best:
        filename = "model_highest_AUC_bag_" + str(bagging + 1) + ".pth.tar"
        torch.save(state, filename)
    else:
        filename = (
            "checkpoint_bag_" + str(bagging + 1) + "_" + str(epoch + 1) + ".pth.tar"
        )
        torch.save(state, filename)


def class_eval(prediction, target, test):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average="binary", zero_division=0
        )
        if not test:
            try:
                auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
            except ValueError:
                auc_score = 0.0
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    if test:
        return accuracy, precision, recall, fscore
    else:
        return accuracy, precision, recall, fscore, auc_score


def optimize_weight(train_loader, model, criterion, optimizer):
    # switch to train mode
    model.train()
    for i, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        if cuda:
            input_var = (item.cuda(non_blocking=True) for item in input)
        else:
            input_var = input
        # normalize target
        target_normed = target.view(-1).long()
        if cuda:
            target_var = target_normed.cuda(non_blocking=True)
        else:
            target_var = target_normed

        output = model(*input_var)
        loss = criterion(output, target_var)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(val_loader, model, criterion, bagging, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    fscores = AverageMeter()
    auc_scores = AverageMeter()

    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()
    end = time.time()
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
        loss = criterion(output, target_var)
        # measure accuracy and record loss
        if test:
            accuracy, precision, recall, fscore = class_eval(
                output.data.cpu(), target, test=True
            )
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            # auc_scores.update(auc_score, target.size(0))
            test_pred = torch.exp(output.data.cpu())
            test_target = target
            assert test_pred.shape[1] == 2
            test_preds += test_pred[:, 1].tolist()
            test_targets += test_target.view(-1).tolist()
            test_cif_ids += batch_cif_ids
        else:
            accuracy, precision, recall, fscore, auc_score = class_eval(
                output.data.cpu(), target, test=False
            )
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print_freq = 10
        # if i % print_freq == 0:
        #     print(
        #         "Test: [{0}/{1}]\t"
        #         "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
        #         "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
        #         "Accu {accu.val:.3f} ({accu.avg:.3f})\t"
        #         "Precision {prec.val:.3f} ({prec.avg:.3f})\t"
        #         "Recall {recall.val:.3f} ({recall.avg:.3f})\t"
        #         "F1 {f1.val:.3f} ({f1.avg:.3f})\t"
        #         "AUC {auc.val:.3f} ({auc.avg:.3f})".format(
        #             i,
        #             len(val_loader),
        #             batch_time=batch_time,
        #             loss=losses,
        #             accu=accuracies,
        #             prec=precisions,
        #             recall=recalls,
        #             f1=fscores,
        #             auc=auc_scores,
        #         )
        #     )

    if test:
        star_label = "**"
        import csv

        with open("test_results_bag_" + str(bagging) + ".csv", "w") as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets, test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = "***"
        print("{star} Loss {losses.avg:.3f}".format(star=star_label, losses=losses))
        print(
            "{star} Recall(TPR) {recall.avg:.3f}".format(
                star=star_label, recall=recalls
            )
        )
        print("{star} AUC {auc.avg:.3f}".format(star=star_label, auc=auc_scores))
    return (auc_scores.avg, recalls.avg, losses.avg)


def train_model(
    split,
    restart=0,
    bag=100,
    workers=5,
    batch_size=256,
    lr=0.01,
    momentum=0.9,
    weight_decay=0,
    lr_milestones=[100],
    gamma=0.1,
    epochs=50,
    user_optim="SGD",
    start_epoch=0,
    pin_memory_flag=True,
    prefetch_factor=10,
):
    # Train/Valid/Test for all bagging loop
    for bagging in range(restart, restart + bag):
        initial_time = time.time()
        collate_fn = collate_batch

        dataset_train = WyckoffData(
            pd.read_csv(
                os.path.join(split, "id_prop_bag_" + str(bagging + 1) + "_train.csv")
            )
        )
        dataset_valid = WyckoffData(
            pd.read_csv(
                os.path.join(split, "id_prop_bag_" + str(bagging + 1) + "_valid.csv")
            )
        )
        train_loader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory_flag,
            prefetch_factor=prefetch_factor,
        )
        val_loader = DataLoader(
            dataset_valid,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory_flag,
            prefetch_factor=prefetch_factor,
        )
        preload_time = time.time() - initial_time
        print("Data loaded in bagging #%d, Time: %f" % (bagging + 1, preload_time))
        # build model
        model = ClassificationModel()
        if cuda:
            model.cuda()
        # define loss func and optimizer
        criterion = nn.NLLLoss()
        if user_optim == "SGD":
            optimizer = optim.SGD(
                model.parameters(), lr, momentum=momentum, weight_decay=weight_decay
            )
        elif user_optim == "Adam":
            optimizer = optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
        elif user_optim == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr, weight_decay=weight_decay)
        else:
            raise NameError("Only SGD or Adam, AdamW is allowed as --optim")

        scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=gamma)
        print("Train/Val/Test in Bagging %d started..." % (bagging + 1))

        model_evaluation_on_training_set = np.zeros((epochs, 3))
        model_evaluation_on_valuation_set = np.zeros((epochs, 3))
        for epoch in range(start_epoch, start_epoch + epochs):
            print(f"current epoch: {epoch}")
            optimize_weight(train_loader, model, criterion, optimizer)
            scheduler.step()
            # evaluate model on validation set and training set
            model_evaluation_on_training_set[epoch - start_epoch, :] = validate(
                train_loader, model, criterion, bagging + 1
            )
            model_evaluation_on_valuation_set[epoch - start_epoch, :] = validate(
                val_loader, model, criterion, bagging + 1
            )
            if (epoch >= (int(epochs * 0.8) - 1)) and (
                int(epochs * 0.8)
                - 1
                + np.argmax(
                    model_evaluation_on_valuation_set[(int(epochs * 0.8) - 1) :, 1]
                )
                == epoch
            ):
                save_checkpoint(
                    {
                        "model_evaluation_on_training_set": model_evaluation_on_training_set,
                        "model_evaluation_on_valuation_set": model_evaluation_on_valuation_set,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    True,
                    bagging,
                    epoch,
                )
        save_checkpoint(
            {
                "model_evaluation_on_training_set": model_evaluation_on_training_set,
                "model_evaluation_on_valuation_set": model_evaluation_on_valuation_set,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            False,
            bagging,
            epoch,
        )
        gc.collect()
