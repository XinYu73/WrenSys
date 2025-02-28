"""
Author: Yu Xin
Date: 2025-02-28 14:03:34
LastEditors: Please set LastEditors
LastEditTime: 2025-02-28 15:52:10
Description: 
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

data_positive = pd.read_csv("unique_experimental.csv")
data_un_label = pd.read_csv("unique_true_theoretical.csv")

random_permutation_positive = np.random.permutation(len(data_positive))
n_train_positive = int(0.8 * len(data_positive))
random_permutation_un_label = np.random.permutation(len(data_un_label))
n_train_un_label = int(0.8 * len(data_un_label))

data_positive_train = data_positive.iloc[random_permutation_positive[:n_train_positive]]
data_positive_evaluation = data_positive.iloc[
    random_permutation_positive[n_train_positive:]
]
data_un_label_train = data_un_label.iloc[random_permutation_un_label[:n_train_un_label]]
data_un_label_evaluation = data_un_label.iloc[
    random_permutation_un_label[n_train_un_label:]
]

train = pd.concat([data_positive_train, data_un_label_train])
train.insert(
    1,
    "synthesizability",
    [1] * len(data_positive_train) + [0] * len(data_un_label_train),
)
train.iloc[:, :2].to_csv("../train/train_set.csv", index=False, header=False)

evaluate = pd.concat([data_positive_evaluation, data_un_label_evaluation])
evaluate.insert(
    1,
    "synthesizability",
    [1] * len(data_positive_evaluation) + [0] * len(data_un_label_evaluation),
)
evaluate.iloc[:, :2].to_csv(
    "../evaluation/evaluation_set.csv", index=False, header=False
)

# train
train = pd.read_csv("../train/train_set.csv", header=None)
names = train.iloc[:, 0].to_list()
synthesizability = train.iloc[:, 1].to_list()
for nn, ss in zip(names, synthesizability):
    if ss == 1:
        if os.path.exists(f"unique_experimental/{nn}.vasp"):
            struct = Structure.from_file(f"unique_experimental/{nn}.vasp")
        else:
            struct = Structure.from_file(f"unique_experimental/{nn}.cif")
    else:
        struct = Structure.from_file(f"unique_theoretical/{nn}.vasp")
    try:
        CifWriter(struct, symprec=0.01, angle_tolerance=5).write_file(
            f"../train/cif_files/{nn}.cif"
        )
    except Exception as e:
        print(nn)


# evaluation
evaluation = pd.read_csv("../evaluation/evaluation_set.csv", header=None)
names = train.iloc[:, 0].to_list()
synthesizability = train.iloc[:, 1].to_list()
for nn, ss in zip(names, synthesizability):
    if ss == 1:
        if os.path.exists(f"unique_experimental/{nn}.vasp"):
            struct = Structure.from_file(f"unique_experimental/{nn}.vasp")
        else:
            struct = Structure.from_file(f"unique_experimental/{nn}.cif")
    else:
        struct = Structure.from_file(f"unique_theoretical/{nn}.vasp")
    try:
        CifWriter(struct, symprec=0.01, angle_tolerance=5).write_file(
            f"../evaluation/cif_files/{nn}.cif"
        )
    except Exception as e:
        print(nn)
