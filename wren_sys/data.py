from __future__ import annotations
import functools
import json
import re
from itertools import groupby
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import LongTensor, Tensor
from torch.utils.data import Dataset
import pickle
from aviary import PKG_DIR
from aviary.wren.utils import relab_dict, wyckoff_multiplicity_dict

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import Structure
from typing import Literal
from string import ascii_uppercase, digits
from itertools import chain, groupby, permutations, product
from os.path import abspath, dirname, join
from pymatgen.core import Composition, Structure
from monty.fractions import gcd
from torch import LongTensor, Tensor
from torch.utils.data import Dataset
from typing import Any, Sequence
import re
import pandas as pd

cry_sys_dict = {
    "triclinic": "a",
    "monoclinic": "m",
    "orthorhombic": "o",
    "tetragonal": "t",
    "trigonal": "h",
    "hexagonal": "h",
    "cubic": "c",
}

cry_param_dict = {
    "a": 6,
    "m": 4,
    "o": 3,
    "t": 2,
    "h": 2,
    "c": 1,
}

remove_digits = str.maketrans("", "", digits)


def get_aflow_label_from_spglib(
    struct: Structure,
    errors: Literal["raise", "annotate", "ignore"] = "ignore",
    symprec=0.1,
    angle_tolerance=5,
) -> str:
    """Get AFLOW prototype label for pymatgen Structure.

    Args:
        struct (Structure): pymatgen Structure object.
        errors ('raise' | 'annotate' | 'ignore']): How to handle errors. 'raise' and
            'ignore' are self-explanatory. 'annotate' prefixes problematic Aflow labels
            with 'invalid <reason>: '.
    Returns:
        str: AFLOW prototype label
    """
    spg_analyzer = SpacegroupAnalyzer(
        struct, symprec=symprec, angle_tolerance=angle_tolerance
    )
    (
        aflow_label_with_chemsys,
        equivalent_wyckoff_labels_number,
    ) = get_aflow_label_from_spg_analyzer(spg_analyzer, errors)

    # try again with refined structure if it initially fails
    # NOTE structures with magmoms fail unless all have same magnetic moment
    if "invalid" in aflow_label_with_chemsys:
        spg_analyzer = SpacegroupAnalyzer(
            spg_analyzer.get_refined_structure(),
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )
        (
            aflow_label_with_chemsys,
            equivalent_wyckoff_labels_number,
        ) = get_aflow_label_from_spg_analyzer(spg_analyzer, errors)

    return aflow_label_with_chemsys, equivalent_wyckoff_labels_number


def get_aflow_label_from_spg_analyzer(
    spg_analyzer: SpacegroupAnalyzer,
    errors: Literal["raise", "annotate", "ignore"] = "raise",
) -> str:
    """Get AFLOW prototype label for pymatgen SpacegroupAnalyzer.

    Args:
        spg_analyzer (SpacegroupAnalyzer): pymatgen SpacegroupAnalyzer object.
        errors ('raise' | 'annotate' | 'ignore']): How to handle errors. 'raise' and
            'ignore' are self-explanatory. 'annotate' prefixes problematic Aflow labels
            with 'invalid <reason>: '.

    Raises:
        ValueError: if errors='raise' and Wyckoff multiplicities do not add up to
            expected composition.

    Raises:
        ValueError: if Wyckoff multiplicities do not add up to expected composition.

    Returns:
        str: AFLOW prototype labels
    """
    spg_num = spg_analyzer.get_space_group_number()
    sym_struct = spg_analyzer.get_symmetrized_structure()

    equivalent_wyckoff_labels = [
        (len(s), s[0].species_string, wyk_letter.translate(remove_digits))
        for s, wyk_letter in zip(
            sym_struct.equivalent_sites, sym_struct.wyckoff_symbols
        )
    ]
    # print(equivalent_wyckoff_labels)
    equivalent_wyckoff_labels = sorted(
        equivalent_wyckoff_labels, key=lambda x: (x[1], x[2])
    )
    equivalent_wyckoff_labels_number = len(equivalent_wyckoff_labels)
    # print(equivalent_wyckoff_labels)
    # check that multiplicities satisfy original composition
    elem_dict = {}
    elem_wyks = []
    for el, g in groupby(
        equivalent_wyckoff_labels, key=lambda x: x[1]
    ):  # sort alphabetically by element
        lg = list(g)
        # print(el, lg)
        elem_dict[el] = sum(
            float(wyckoff_multiplicity_dict[str(spg_num)][e[2]]) for e in lg
        )
        wyks = ""
        for wyk, w in groupby(
            lg, key=lambda x: x[2]
        ):  # sort alphabetically by wyckoff letter
            lw = list(w)
            wyks += f"{len(lw)}{wyk}"
        elem_wyks.append(wyks)

    # canonicalize the possible wyckoff letter sequences
    canonical = canonicalize_elem_wyks("_".join(elem_wyks), spg_num)

    # get Pearson symbol
    cry_sys = spg_analyzer.get_crystal_system()
    # print(cry_sys)
    spg_sym = spg_analyzer.get_space_group_symbol()
    # print("spg_sym", spg_sym)
    centering = "C" if spg_sym[0] in ("A", "B", "C", "S") else spg_sym[0]
    num_sites_conventional = len(spg_analyzer.get_symmetry_dataset()["std_types"])
    pearson_symbol = f"{cry_sys_dict[cry_sys]}{centering}{num_sites_conventional}"

    prototype_form = prototype_formula(sym_struct.composition)
    # print("prototype_form", prototype_form)
    chem_sys = sym_struct.composition.chemical_system
    aflow_label_with_chemsys = (
        f"{prototype_form}_{pearson_symbol}_{spg_num}_{canonical}:{chem_sys}"
    )

    observed_formula = Composition(elem_dict).reduced_formula
    expected_formula = sym_struct.composition.reduced_formula
    if observed_formula != expected_formula:
        if errors == "raise":
            raise ValueError(
                f"Invalid WP multiplicities - {aflow_label_with_chemsys}, expected "
                f"{observed_formula} to be {expected_formula}"
            )
        elif errors == "annotate":
            return f"invalid multiplicities: {aflow_label_with_chemsys}"

    return aflow_label_with_chemsys, equivalent_wyckoff_labels_number


def canonicalize_elem_wyks(elem_wyks: str, spg_num: int) -> str:
    """Given an element ordering, canonicalize the associated Wyckoff positions
    based on the alphabetical weight of equivalent choices of origin.

    Args:
        elem_wyks (str): Wren Wyckoff string encoding element types at Wyckoff positions
        spg_num (int): International space group number.

    Returns:
        str: Canonicalized Wren Wyckoff encoding.
    """
    isopointal = []
    # !!!! important
    for trans in relab_dict[str(spg_num)]:
        t = str.maketrans(trans)
        isopointal.append(elem_wyks.translate(t))

    isopointal = list(set(isopointal))

    scores = []
    sorted_iso = []
    for wyks in isopointal:
        sorted_el_wyks, score = sort_and_score_wyks(wyks)
        scores.append(score)
        sorted_iso.append(sorted_el_wyks)
        # print(
        "sorted(zip(scores, sorted_iso), key=lambda x: (x[0], x[1]))[0][1]",
    #    sorted(zip(scores, sorted_iso), key=lambda x: (x[0], x[1])),)
    # print(zip(scores, sorted_iso))
    canonical = sorted(zip(scores, sorted_iso), key=lambda x: (x[0], x[1]))[0][1]

    return canonical


def sort_and_score_wyks(wyks: str) -> tuple[str, int]:
    """Determines the order or Wyckoff positions when canonicalizing Aflow labels.

    Args:
        wyks (str): Wyckoff position substring from AFLOW-style prototype label

    Returns:
        tuple: containing
        - str: sorted Wyckoff position substring for AFLOW-style prototype label
        - int: integer score to rank order when canonicalizing
    """
    # print("wyks", wyks)
    score = 0
    sorted_el_wyks = []
    for el_wyks in wyks.split("_"):
        sep_el_wyks = ["".join(g) for _, g in groupby(el_wyks, str.isalpha)]
        # print("sep_el_wyks1", sep_el_wyks)
        sep_el_wyks = ["" if i == "1" else i for i in sep_el_wyks]
        # print("sep_el_wyks2", sep_el_wyks)
        sorted_el_wyks.append(
            "".join(
                [
                    f"{n}{w}"
                    for n, w in sorted(
                        zip(sep_el_wyks[0::2], sep_el_wyks[1::2]),
                        key=lambda x: x[1],
                    )
                ]
            )
        )
        # print("########", [ord(el) - 96 for el in sep_el_wyks[1::2]])
        score += sum(0 if el == "A" else ord(el) - 96 for el in sep_el_wyks[1::2])

    return "_".join(sorted_el_wyks), score


def prototype_formula(composition: Composition) -> str:
    """An anonymized formula. Unique species are arranged in alphabetical order
    and assigned ascending alphabets. This format is used in the aflow structure
    prototype labelling scheme.

    Args:
        composition (Composition): Pymatgen Composition to process

    Returns:
        str: anonymized formula where the species are in alphabetical order
    """
    reduced = composition.element_composition
    if all(x == int(x) for x in composition.values()):
        reduced /= gcd(*(int(i) for i in composition.values()))

    amounts = [reduced[key] for key in sorted(reduced, key=str)]

    anon = ""
    for e, amt in zip(ascii_uppercase, amounts):
        if amt == 1:
            amt_str = ""
        elif abs(amt % 1) < 1e-8:
            amt_str = str(int(amt))
        else:
            amt_str = str(amt)
        anon += f"{e}{amt_str}"
    return anon


class WyckoffData(Dataset):
    """Wyckoff dataset class for the Wren model."""

    def __init__(
        self,
        df: pd.DataFrame,
        elem_embedding: str = "matscholar200",
        sym_emb: str = "bra-alg-off",
        inputs: str = "wyckoff",
        identifiers: str = "mpID",
    ):
        """Data class for Wren models.
        Args:
            df (pd.DataFrame): Pandas dataframe holding input and target values.
            task_dict (dict[str, "regression" | "classification"]): Map from target names to task
                type for multi-task learning.
            elem_embedding (str, optional): One of "matscholar200", "cgcnn92", "megnet16",
                "onehot112" or path to a file with custom element embeddings.
                Defaults to "matscholar200".
            sym_emb (str): Symmetry embedding. One of "bra-alg-off" (default) or "spg-alg-off".
            inputs (str, optional): df columns to be used for featurisation.
                Defaults to "wyckoff".
            identifiers (list, optional): df columns for distinguishing data points. Will be
                copied over into the model's output CSV. Defaults to ["material_id", "composition"].
        """
        self.inputs = inputs
        self.identifiers = identifiers
        self.df = df
        if elem_embedding in ["matscholar200", "cgcnn92", "megnet16", "onehot112"]:
            elem_embedding = f"{PKG_DIR}/embeddings/element/{elem_embedding}.json"
        with open(elem_embedding) as emb_file:
            self.elem_features = json.load(emb_file)
        self.elem_emb_len = len(list(self.elem_features.values())[0])
        if sym_emb in ["bra-alg-off", "spg-alg-off"]:
            sym_emb = f"{PKG_DIR}/embeddings/wyckoff/{sym_emb}.json"
        with open(sym_emb) as sym_file:
            self.sym_features = json.load(sym_file)
        self.sym_emb_len = len(list(list(self.sym_features.values())[0].values())[0])

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        df_repr = f"cols=[{', '.join(self.df.columns)}], len={len(self.df)}"
        return f"{type(self).__name__}({df_repr}, task_dict={self.task_dict})"

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx: int):
        """Get an entry out of the Dataset
        Args:
            idx (int): index of entry in Dataset
        Returns:
            tuple containing:
            - tuple[Tensor, Tensor, Tensor, LongTensor, LongTensor]: Wren model inputs
            - list[Tensor | LongTensor]: regression or classification targets
            - list[str | int]: identifiers like material_id, composition
        """
        row = self.df.iloc[idx]
        wyckoff_str = row[self.inputs]
        material_ids = row[self.identifiers]
        parsed_output = parse_aflow_wyckoff_str(wyckoff_str)
        spg_num, wyk_site_multiplcities, elements, augmented_wyks = parsed_output
        wyk_site_multiplcities = np.atleast_2d(wyk_site_multiplcities).T / np.sum(
            wyk_site_multiplcities
        )
        try:
            element_features = np.vstack([self.elem_features[el] for el in elements])
        except AssertionError:
            print(f"Failed to process elements for {material_ids}")
            raise
        try:
            symmetry_features = np.vstack(
                [
                    self.sym_features[spg_num][wyk_site]
                    for wyckoff_sites in augmented_wyks
                    for wyk_site in wyckoff_sites
                ]
            )
        except AssertionError:
            print(f"Failed to process Wyckoff positions for {material_ids}")
            raise
        n_wyks = len(elements)
        self_idx = []
        nbr_idx = []
        for i in range(n_wyks):
            self_idx += [i] * n_wyks
            nbr_idx += list(range(n_wyks))
        self_aug_fea_idx = []
        nbr_aug_fea_idx = []
        n_aug = len(augmented_wyks)
        for i in range(n_aug):
            self_aug_fea_idx += [x + i * n_wyks for x in self_idx]
            nbr_aug_fea_idx += [x + i * n_wyks for x in nbr_idx]
        # convert all data to tensors
        wyckoff_weights = Tensor(wyk_site_multiplcities)
        element_features = Tensor(element_features)
        symmetry_features = Tensor(symmetry_features)
        self_idx = LongTensor(self_aug_fea_idx)
        nbr_idx = LongTensor(nbr_aug_fea_idx)
        targets = torch.Tensor([float(self.df.iloc[idx][-1])])
        return (
            (wyckoff_weights, element_features, symmetry_features, self_idx, nbr_idx),
            targets,
            material_ids,
        )


def parse_aflow_wyckoff_str(
    aflow_label: str,
) -> tuple[str, list[float], list[str], list[tuple[str, ...]]]:
    """Parse the Wren AFLOW-like Wyckoff encoding.

    Args:
        aflow_label (str): AFLOW-style prototype string with appended chemical system

    Returns:
        tuple[str, list[float], list[str], list[str]]: spacegroup number, Wyckoff site
            multiplicities, elements symbols and equivalent wyckoff sets
    """
    proto, chemsys = aflow_label.split(":")
    elems = chemsys.split("-")
    _, _, spg_num, *wyckoff_letters = proto.split("_")
    wyckoff_site_multiplicities = []
    elements = []
    wyckoff_set = []
    for el, wyk_letters_per_elem in zip(elems, wyckoff_letters):
        # normalize Wyckoff letters to start with 1 if missing digit
        wyk_letters_per_elem = re.sub(
            r"((?<![0-9])[A-z])", r"1\g<1>", wyk_letters_per_elem
        )
        # Separate out pairs of Wyckoff letters and their number of occurrences
        sep_n_wyks = ["".join(g) for _, g in groupby(wyk_letters_per_elem, str.isalpha)]
        for n, l in zip(sep_n_wyks[0::2], sep_n_wyks[1::2]):
            m = int(n)
            elements.extend([el] * m)
            wyckoff_set.extend([l] * m)
            wyckoff_site_multiplicities.extend(
                [float(wyckoff_multiplicity_dict[spg_num][l])] * m
            )
    # NOTE This on-the-fly augmentation of equivalent Wyckoff sets could be a source of high
    # memory use. Can be turned off by commenting out the for loop and returning
    # [wyckoff_set] instead of augmented_wyckoff_set. Wren should be able to learn anyway.
    augmented_wyckoff_set = []
    for trans in relab_dict[spg_num]:
        # Apply translation dictionary of allowed relabelling operations in spacegroup
        t = str.maketrans(trans)
        augmented_wyckoff_set.append(
            tuple(",".join(wyckoff_set).translate(t).split(","))
        )
    augmented_wyckoff_set = list(set(augmented_wyckoff_set))
    return spg_num, wyckoff_site_multiplicities, elements, augmented_wyckoff_set
    # return spg_num, wyckoff_site_multiplicities, elements, [wyckoff_set]


def collate_batch(
    samples: tuple[
        tuple[Tensor, Tensor, Tensor, LongTensor, LongTensor],
        list[Tensor | LongTensor],
        list[str | int],
    ],
) -> tuple[Any, ...]:
    """Collate a list of data and return a batch for predicting
    crystal properties.
    Args:
        samples ([tuple]): list of tuples for each data point.
            (elem_fea, nbr_fea, nbr_idx, target)
            elem_fea (Tensor): Node features from atom type and Wyckoff letter
            nbr_fea (Tensor): _description_
            nbr_idx (LongTensor):
            target (Tensor):
            cif_id: str or int
    Returns:
        tuple[
            tuple[Tensor, Tensor, Tensor, LongTensor, LongTensor, LongTensor, LongTensor]:
                batched Wren model inputs,
            tuple[Tensor | LongTensor]: Target values for different tasks,
            *tuple[str | int]]: Identifiers like material_id, composition
        ]
    """
    # define the lists
    batch_mult_weights = []
    batch_elem_fea = []
    batch_sym_fea = []
    batch_self_idx = []
    batch_nbr_idx = []
    crystal_wyk_idx = []
    aug_cry_idx = []
    batch_targets = []
    batch_cry_ids = []

    aug_count = 0
    cry_base_idx = 0
    for idx, (inputs, target, *cry_ids) in enumerate(samples):
        mult_weights, elem_fea, sym_fea, self_idx, nbr_idx = inputs
        n_elem = elem_fea.shape[0]
        n_sites = sym_fea.shape[0]  # number of atoms for this crystal
        n_aug = int(float(n_sites) / float(n_elem))
        # batch the features together
        batch_mult_weights.append(mult_weights.repeat((n_aug, 1)))
        batch_elem_fea.append(elem_fea.repeat((n_aug, 1)))
        batch_sym_fea.append(sym_fea)
        # mappings from bonds to atoms
        batch_self_idx.append(self_idx + cry_base_idx)
        batch_nbr_idx.append(nbr_idx + cry_base_idx)
        # mapping from atoms to crystals
        crystal_wyk_idx.append(
            torch.tensor(range(aug_count, aug_count + n_aug)).repeat_interleave(n_elem)
        )
        aug_cry_idx.append(torch.tensor([idx] * n_aug))
        # batch the targets and ids
        batch_targets.append(target)
        batch_cry_ids.append(cry_ids)
        # increment the id counter
        aug_count += n_aug
        cry_base_idx += n_sites

    return (
        (
            torch.cat(batch_mult_weights, dim=0),
            torch.cat(batch_elem_fea, dim=0),
            torch.cat(batch_sym_fea, dim=0),
            torch.cat(batch_self_idx, dim=0),
            torch.cat(batch_nbr_idx, dim=0),
            torch.cat(crystal_wyk_idx),
            torch.cat(aug_cry_idx),
        ),
        torch.stack(batch_targets, dim=0),
        *zip(*batch_cry_ids),
    )


def preprocess(pickle_folder, pdframe_path):
    datacsv = pd.read_csv(pdframe_path)
    dataset = WyckoffData(datacsv)
    for i in range(len(datacsv)):
        with open(pickle_folder + "/" + str(datacsv["mpID"][i]) + ".pickle", "wb") as f:
            pickle.dump(dataset[i], f)
