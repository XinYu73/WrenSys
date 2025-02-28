"""
Author: Yu Xin
Date: 2025-02-28 13:43:52
LastEditors: Please set LastEditors
LastEditTime: 2025-02-28 15:38:48
Description: 
"""

import numpy as np
import pandas as pd
import os
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher


data_positive = pd.read_csv("unique_experimental.csv")
data_un_label = pd.read_csv("unique_theoretical.csv")

un_label_index_to_drop = []
for index, row in data_un_label.iterrows():
    sub_data_positive = data_positive.loc[
        (data_positive["group_number"] == row["group_number"])
        & (data_positive["compositions"] == row["compositions"])
    ]
    the_struct = Structure.from_file(f'unique_theoretical/{row["name"]}.vasp')
    for index2, row2 in sub_data_positive.iterrows():
        if os.path.exists(f'unique_experimental/{row2["name"]}.vasp'):
            exp_struct = Structure.from_file(f'unique_experimental/{row2["name"]}.vasp')
        else:
            exp_struct = Structure.from_file(f'unique_experimental/{row2["name"]}.cif')
        sm = StructureMatcher()
        if sm.fit(the_struct, exp_struct):
            un_label_index_to_drop.append(index)
            break

data_un_label.drop(data_un_label.index[un_label_index_to_drop], inplace=True)
data_un_label.to_csv("unique_true_theoretical.csv")
