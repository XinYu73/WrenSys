"""
Author: Yu Xin
Date: 2025-02-26 05:35:29
LastEditors: Please set LastEditors
LastEditTime: 2025-02-26 05:47:36
Description: 
"""

"""
Author: Yu Xin
Date: 2025-02-26 05:35:29
LastEditors: Please set LastEditors
LastEditTime: 2025-02-26 05:38:55
Description: 
"""

import os
import pandas as pd
from pymatgen.core import Structure
from wren_sys.data import get_aflow_label_from_spglib

mpID = []
wyckoff = []
equivalent_wyckoff_labels = []
number_of_atoms = []
synthesisability = []
for item in os.listdir("experimental"):
    if item.startswith("icsd"):
        try:
            struct = Structure.from_file("experimental/" + item)
            encode, equivalent_wyckoff_labels_number = get_aflow_label_from_spglib(
                struct, symprec=0.1, angle_tolerance=5
            )
            wyckoff.append(encode)
            equivalent_wyckoff_labels.append(equivalent_wyckoff_labels_number)
            number_of_atoms.append(int(encode.split("_")[1][2:]))
            mpID.append(item.split(".")[0])
            synthesisability.append(1)
        except Exception as e:
            print(e)
df = pd.DataFrame(
    {
        "mpID": mpID,
        "wyckoff": wyckoff,
        "number_of_atoms": number_of_atoms,
        "equivalent_wyckoff_labels_number": equivalent_wyckoff_labels,
        "synthesisability": synthesisability,
    }
).to_csv("icsd_wyckoff.csv", index=False)
