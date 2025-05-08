"""
Author: Yu Xin
Date: 2025-04-08 15:48:59
LastEditors: Please set LastEditors
LastEditTime: 2025-04-08 16:32:53
Description: 
"""

import os
import pickle
import argparse
import multiprocessing

import numpy as np
import pandas as pd

from ase.db import connect
from ase.io import read, write
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure

aaa = AseAtomsAdaptor()

from wren_sys.data import get_aflow_label_from_spglib


def structure_loader(path, name_tag):
    if path.endswith(".db"):
        db = connect(path)
        for row in db.select():
            yield (
                "_".join(row.key_value_pairs[name_tag].split("-")),
                aaa.get_structure(row.toatoms()),
            )
    else:
        for item in os.listdir(path):
            if item.endswith(".vasp") or item.endswith(".cif"):
                yield (
                    item.split(".")[0],
                    aaa.get_structure(read(os.path.join(path, item))),
                )


class DataStore:
    def __init__(self):
        self.mpID = []
        self.wyckoff = []
        self.number_of_atoms = []
        self.equivalent_wyckoff_labels_number = []
        self.targets = []

    def update(self, one_pack: None | dict):
        if one_pack is not None:
            self.mpID.append(one_pack["mpID"])
            self.wyckoff.append(one_pack["wyckoff"])
            self.number_of_atoms.append(one_pack["number_of_atoms"])
            self.equivalent_wyckoff_labels_number.append(
                one_pack["equivalent_wyckoff_labels_number"]
            )
            self.targets.append(0)

    def to_dataframe(self, csv_path: str):
        df = pd.DataFrame(
            {
                "mpID": self.mpID,
                "wyckoff": self.wyckoff,
                "number_of_atoms": self.number_of_atoms,
                "equivalent_wyckoff_labels_number": self.equivalent_wyckoff_labels_number,
                "target": self.targets,
            }
        )
        df.to_csv(csv_path, index=False)


def one_job(mpID: str, struct: Structure, symprec: float, angle_tolerance: float):
    try:
        (
            aflow_label_with_chemsys,
            equivalent_wyckoff_labels_number,
        ) = get_aflow_label_from_spglib(
            struct=struct, symprec=symprec, angle_tolerance=angle_tolerance
        )
        return {
            "mpID": mpID,
            "wyckoff": aflow_label_with_chemsys,
            "number_of_atoms": int(aflow_label_with_chemsys.split("_")[1][2:]),
            "equivalent_wyckoff_labels_number": equivalent_wyckoff_labels_number,
        }
    except Exception as e:
        print(f"Error in get_aflow_label_from_spglib: {e}")
        return None


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--structure_path", help="structure path, or db", default="structure_path"
    )
    parser.add_argument(
        "--name_tag", help="only for db, the key for name", default="name"
    )
    parser.add_argument(
        "--max_process",
        help="maximum number of process to be used",
        default=1,
        type=int,
    )
    parser.add_argument("--output_csv", help="output csv", default="to_network.csv")

    args = parser.parse_args()
    ds = DataStore()
    pool = multiprocessing.Pool(processes=args.max_process)
    result = []
    for one_name, one_atoms in structure_loader(args.structure_path, args.name_tag):
        result.append(
            pool.apply_async(
                one_job,
                (
                    one_name,
                    one_atoms,
                    0.1,  # symprec
                    5.0,  # angle_tolerance
                ),
            )
        )
    pool.close()
    pool.join()

    for rr in result:
        ds.update(rr.get())
    ds.to_dataframe(args.output_csv)
