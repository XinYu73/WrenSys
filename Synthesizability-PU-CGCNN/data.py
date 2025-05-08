"""
Author: Yu Xin
Date: 2025-05-08 08:32:59
LastEditors: Please set LastEditors
LastEditTime: 2025-05-08 08:39:04
Description: 
"""

import csv
import pickle
from torch.utils.data import Dataset


class CGCNNData(Dataset):
    def __init__(self, graph_dir, id_prop_file="id_prop.csv"):
        self.graph_dir = graph_dir
        with open(id_prop_file) as g:
            reader = csv.reader(g)
            self.cif_list = [row[0] for row in reader]

    def __len__(self) -> int:
        return len(self.cif_list)

    def __getitem__(self, index):
        print(self.cif_list[index])
        with open(self.graph_dir + "/" + self.cif_list[index] + ".pickle", "rb") as f:
            return pickle.load(f)
