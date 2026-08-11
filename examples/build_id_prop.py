"""
Author: Yu Xin
Date: 2024-12-02 07:46:41
LastEditors: Please set LastEditors
LastEditTime: 2024-12-02 07:46:42
Description:
"""

import os

with open("./cif_files/id_prop.csv", "w") as f:
    for file in os.listdir("./cif_files"):
        if file.endswith(".cif"):
            f.writelines(f"{file.split('.')[0]},{0}\n")

