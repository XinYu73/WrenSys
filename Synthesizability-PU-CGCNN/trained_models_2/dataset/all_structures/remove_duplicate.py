"""
Author: Yu Xin
Date: 2025-02-25 15:46:20
LastEditors: Please set LastEditors
LastEditTime: 2025-02-25 16:22:10
Description: 
"""

import os
import importlib
import pickle
import argparse
import multiprocessing
from ase.db import connect
import pandas as pd
import numpy as np
from pymatgen.core import Structure, Composition, Element
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.phase_diagram import PhaseDiagram


from ase.io import read
from sepibs.data_tool import LayerGroupAnalyzer
from sepibs.structure_relaxation.filter_result import (
    aaa,
    PDEntry,
    one_job,
    load_from_existed,
    configuration_check,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original_structure_path",
        help="structure path, absolute path",
        default="../1_ml_opted_low_energy_unique",
    )
    parser.add_argument(
        "--unique_structure_path",
        help="structure path, absolute path",
        default="../1_ml_opted_low_energy_unique",
    )
    args = parser.parse_args()

    group_numbers = []
    compositions = []
    structures = []
    names = []
    optimization_folder = args.original_structure_path
    max_process = 12
    dim = 3

    for item in [
        file
        for file in os.listdir(optimization_folder)
        if file.endswith(".cif") or file.endswith(".vasp")
    ]:
        struct = aaa.get_structure(read(os.path.join(optimization_folder, item)))
        try:
            if dim == 3:
                sa = SpacegroupAnalyzer(struct, symprec=0.01, angle_tolerance=5)
                group_numbers.append(sa.get_space_group_number())
                structures.append(sa.get_primitive_standard_structure())
        except Exception as e:
            print(f"{e} occur in select_unique_ones at {item}")
            group_numbers.append(1)
            structures.append(struct)

        compositions.append(structures[-1].composition.formula)
        names.append(item.split(".")[0])

    work_data = pd.DataFrame(
        {
            "name": names,
            "group_number": group_numbers,
            "compositions": compositions,
            "energy_per_atoms": [0] * len(group_numbers),
        }
    )

    pool = multiprocessing.Pool(processes=max_process)
    for select_group_number in set(group_numbers):
        for select_composition in set(compositions):
            selected_work_data = work_data.loc[
                (work_data.group_number == select_group_number)
                & (work_data.compositions == select_composition)
            ]
            pool.apply_async(
                one_job,
                (
                    [structures[ind] for ind in selected_work_data.index],
                    selected_work_data,
                    args.unique_structure_path,
                    [],
                ),
            )
    pool.close()
    pool.join()

    work_data.loc[
        work_data.name.apply(
            lambda x: x + ".vasp" in os.listdir(args.unique_structure_path)
            or x + ".cif" in os.listdir(args.unique_structure_path)
        )
    ].to_csv(
        f"{args.unique_structure_path}/{os.path.basename(args.unique_structure_path)}.csv",
        index=None,
    )
