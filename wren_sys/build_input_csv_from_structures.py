"""
Author: Yu Xin
Date: 2025-04-08 15:48:59
LastEditors: Please set LastEditors
LastEditTime: 2025-04-08 16:32:53
Description: 
"""

import os
import argparse
import multiprocessing
from itertools import islice

import numpy as np
import pandas as pd

from ase.db import connect
from ase.io import read, write
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure

aaa = AseAtomsAdaptor()

from wren_sys.data import get_aflow_label_from_spglib


# def structure_loader(path, name_tag):
#     if path.endswith(".db"):
#         db = connect(path)
#         for row in db.select():
#             yield (
#                 "_".join(row.key_value_pairs[name_tag].split("-")),
#                 aaa.get_structure(row.toatoms()),
#             )
#     else:
#         for item in os.listdir(path):
#             if item.endswith(".vasp") or item.endswith(".cif"):
#                 yield (
#                     item.split(".")[0],
#                     aaa.get_structure(read(os.path.join(path, item))),
#                 )


def split_iterable(iterable, n):
    it = iter(iterable)
    length = sum(1 for _ in iterable)
    avg, remainder = divmod(length, n)
    sizes = [avg + (1 if i < remainder else 0) for i in range(n)]

    for size in sizes:
        yield list(islice(it, size))


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

    def to_dataframe(self):
        df = pd.DataFrame(
            {
                "mpID": self.mpID,
                "wyckoff": self.wyckoff,
                "number_of_atoms": self.number_of_atoms,
                "equivalent_wyckoff_labels_number": self.equivalent_wyckoff_labels_number,
                "target": self.targets,
            }
        )
        # df.to_csv(csv_path, index=False)
        return df


def one_job(
    structure_path: str, name_list: list, symprec: float, angle_tolerance: float
)->None|pd.DataFrame:
    ds = DataStore()
    for item in name_list:
        try:
            (
                aflow_label_with_chemsys,
                equivalent_wyckoff_labels_number,
            ) = get_aflow_label_from_spglib(
                struct=Structure.from_file(os.path.join(structure_path, item)),
                symprec=symprec,
                angle_tolerance=angle_tolerance,
            )
            one_pack = {
                "mpID": item.split(".")[0],
                "wyckoff": aflow_label_with_chemsys,
                "number_of_atoms": int(aflow_label_with_chemsys.split("_")[1][2:]),
                "equivalent_wyckoff_labels_number": equivalent_wyckoff_labels_number,
            }
        except Exception as e:
            print(f"Error in get_aflow_label_from_spglib: {item}")
            # return None
            one_pack = None

        ds.update(one_pack=one_pack)
    if len(ds.mpID) > 0:
        return ds.to_dataframe()
    else:
        return None


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--structure_path", help="structure path", default="structure_path"
    )
    # parser.add_argument(
    #     "--name_tag", help="only for db, the key for name", default="name"
    # )
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
    structure_names = list(
        filter(
            lambda x: (x.endswith(".vasp")) or (x.endswith(".cif")),
            os.listdir(args.structure_path),
        )
    )
    for sub_list in split_iterable(structure_names, args.max_process):
        result.append(
            pool.apply_async(
                one_job,
                (
                    args.structure_path,
                    sub_list,
                    0.1,  # symprec
                    5.0,  # angle_tolerance
                ),
            )
        )
    pool.close()
    pool.join()

    result_pd = []
    for rr in result:
        out = rr.get()
        if out is not None:
            result_pd.append(out)

    pd.concat(result_pd).to_csv(args.output_csv, index=False)
