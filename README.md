# Wren_Sys

This repository contains the machine-learning part of the work associated with
https://link.aps.org/doi/10.1103/6t5z-tqym.

It provides two complementary synthesizability evaluation pipelines:

1. **Wren_Sys / Wyckoff encoding model**: a Wren-style existence/synthesizability
   evaluation model based on AFLOW-like Wyckoff encodings. Code lives in
   `wren_sys/`, and pretrained bagging models are in `model_dir/` and
   `model_dir_2/`.
2. **CLScore / PU-CGCNN model**: a crystal-likeness score evaluation pipeline
   based on the Synthesizability-PU-CGCNN code. Code lives in
   `Synthesizability-PU-CGCNN/`.

`Synthesizability-PU-CGCNN/` was copied from another repository and modified for
this project. Its original README is preserved at
`Synthesizability-PU-CGCNN/README.md`; please also cite the original
Synthesizability-PU-CGCNN / CLScore work if you use that part of the code.

## Repository Layout

```text
.
|-- wren_sys/                         # Wyckoff-encoding Wren_Sys model
|   |-- build_input_csv_from_structures.py
|   |-- data.py
|   |-- model.py
|   |-- predict.py
|   `-- train.py
|-- model_dir/                        # 56 pretrained Wren_Sys bagging models
|-- model_dir_2/                      # another 56-model Wren_Sys checkpoint set
|-- Synthesizability-PU-CGCNN/         # modified CLScore / PU-CGCNN code
|   |-- generate_crystal_graph.py
|   |-- predict_PU_learning.py
|   |-- main_PU_learning.py
|   |-- trained_models/               # 100 pretrained PU-CGCNN models
|   `-- trained_models_5/             # another 100-model checkpoint set
`-- examples/gnome_part/              # small example input/output files
```

## Installation

Create a Python environment first. The code has been used with Python 3 and
PyTorch.

```bash
git clone https://github.com/XinYu73/Wren_Sys.git
cd Wren_Sys
pip install -e .
```

Install the scientific Python and materials packages required by the scripts:

```bash
pip install numpy pandas scikit-learn tqdm pymatgen ase
pip install torch
pip install aviary torch-scatter
```

`torch-scatter` is sensitive to the installed PyTorch and CUDA versions. If the
plain `pip install torch-scatter` command fails, install the wheel matching your
local PyTorch/CUDA environment from the PyTorch Geometric wheel index.

## Input Files

### Wren_Sys input CSV

The Wren_Sys predictor reads a CSV file with these columns:

```csv
mpID,wyckoff,number_of_atoms,equivalent_wyckoff_labels_number,target
0347f8a748,A4BC13_hP18_156_a2bc_a_4a4b5c:Hg-In-Li,18,18,0
```

Column meanings:

- `mpID`: unique structure identifier.
- `wyckoff`: AFLOW-like Wyckoff encoding with chemical system appended after `:`.
- `number_of_atoms`: number of atoms in the conventional cell used by the
  encoding.
- `equivalent_wyckoff_labels_number`: number of equivalent Wyckoff labels.
- `target`: dummy or label column. For prediction-only use, set it to `0`.

You can generate this CSV directly from `.cif` or `.vasp` structures:

```bash
python wren_sys/build_input_csv_from_structures.py \
  --structure_path examples/gnome_part/selected_structures \
  --max_process 2 \
  --output_csv input_wren_sys.csv
```

### CLScore / PU-CGCNN input folder

The CLScore pipeline expects a folder containing:

```text
cif_files/
|-- atom_init.json
|-- id_prop.csv
|-- ID_1.cif
|-- ID_2.cif
`-- ...
```

`id_prop.csv` has no header and contains:

```csv
ID_1,0
ID_2,0
```

For prediction-only use, set the second column to `0`. The example helper writes
this file from all `.cif` files in `./cif_files`:

```bash
cd examples/gnome_part
cp ../../Synthesizability-PU-CGCNN/trained_models/atom_init.json cif_files/
python ../build_id_prop.py
```

## Run Wren_Sys Prediction

You can directly use the prepared example in `examples/gnome_part/` to test the
calculation. From the repository root:

```bash
python wren_sys/predict.py \
  -m model_dir \
  -p examples/gnome_part/input_wren_sys.csv \
  -n examples/gnome_part/output_wren_sys.csv
```

The output CSV is sorted by the predicted `CLscore` column in descending order
and keeps the original Wyckoff features:

```csv
mpID,CLscore,wyckoff,number_of_atoms,equivalent_wyckoff_labels_number,target
```

Note: `wren_sys/predict.py` currently uses `bag=56` internally, so the model
directory should contain checkpoints named:

```text
model_highest_AUC_bag_1.pth.tar
...
model_highest_AUC_bag_56.pth.tar
```

## Run CLScore / PU-CGCNN Prediction

The PU-CGCNN pipeline first converts `.cif` structures into crystal-graph pickle
files and then evaluates the 100-model bagging ensemble.

The prepared `examples/gnome_part/` folder already contains example CIF files,
`id_prop.csv`, `atom_init.json`, and example outputs, so it can be used directly
as a test case:

```bash
cd examples/gnome_part

python ../../Synthesizability-PU-CGCNN/generate_crystal_graph.py \
  --cifs ./cif_files \
  --n 12 \
  --r 8 \
  --f ./saved_crystal_graph

python ../../Synthesizability-PU-CGCNN/predict_PU_learning.py \
  --bag 100 \
  --graph ./saved_crystal_graph \
  --cifs ./cif_files \
  --modeldir ../../Synthesizability-PU-CGCNN/trained_models
```

The ensemble output is written in the current working directory:

```text
test_results_ensemble_100models.csv
```

with columns:

```csv
id,CLscore,bagging
```

## Example Workflow

The directory `examples/gnome_part/` is the recommended starting point for
checking that the two workflows run correctly. It contains three example
structures, Wren_Sys input/output files, CLScore input files, generated crystal
graphs, and example prediction outputs.

You can run the two example calculations with:

```bash
# Wren_Sys example, from the repository root
python wren_sys/predict.py \
  -m model_dir \
  -p examples/gnome_part/input_wren_sys.csv \
  -n examples/gnome_part/output_wren_sys.csv

# CLScore / PU-CGCNN example
cd examples/gnome_part
python ../../Synthesizability-PU-CGCNN/generate_crystal_graph.py \
  --cifs ./cif_files \
  --n 12 \
  --r 8 \
  --f ./saved_crystal_graph
python ../../Synthesizability-PU-CGCNN/predict_PU_learning.py \
  --bag 100 \
  --graph ./saved_crystal_graph \
  --cifs ./cif_files \
  --modeldir ../../Synthesizability-PU-CGCNN/trained_models
```

The same commands are also collected in:

```text
examples/gnome_part/wren_sys.sh
examples/gnome_part/clscore.sh
examples/command_log
```

For a new structure set, a typical workflow is:

```bash
# 1. Prepare Wren_Sys input from .cif or .vasp files
python wren_sys/build_input_csv_from_structures.py \
  --structure_path path/to/structures \
  --max_process 4 \
  --output_csv input_wren_sys.csv

# 2. Predict with the Wyckoff-encoding model
python wren_sys/predict.py \
  -m model_dir \
  -p input_wren_sys.csv \
  -n output_wren_sys.csv

# 3. Prepare CLScore input folder
mkdir -p cif_files
# Put .cif files into cif_files/ and add atom_init.json.
# Then create cif_files/id_prop.csv as ID,0 rows.

# 4. Build CGCNN crystal graphs and predict CLScore
python Synthesizability-PU-CGCNN/generate_crystal_graph.py \
  --cifs ./cif_files \
  --n 12 \
  --r 8 \
  --f ./saved_crystal_graph

python Synthesizability-PU-CGCNN/predict_PU_learning.py \
  --bag 100 \
  --graph ./saved_crystal_graph \
  --cifs ./cif_files \
  --modeldir Synthesizability-PU-CGCNN/trained_models
```

## Training Notes

### Wren_Sys

`wren_sys/train.py` exposes the function `train_model(...)`. It expects a split
directory containing files like:

```text
id_prop_bag_1_train.csv
id_prop_bag_1_valid.csv
...
```

Each split CSV should have the same columns as the Wren_Sys input CSV. The best
checkpoint for each bag is saved as:

```text
model_highest_AUC_bag_N.pth.tar
```

### CLScore / PU-CGCNN

The modified PU-CGCNN training entry point is:

```bash
python Synthesizability-PU-CGCNN/main_PU_learning.py \
  --bag 100 \
  --graph ./saved_crystal_graph \
  --cifs ./cif_files \
  --split ./split
```

This follows the original Synthesizability-PU-CGCNN workflow: generate crystal
graphs, split positive/unlabeled data for bagging, train one CGCNN classifier per
bag, and aggregate predictions.

## Citation and Attribution

If you use this repository, please cite the related work at:

- https://link.aps.org/doi/10.1103/6t5z-tqym

For the CLScore / PU-CGCNN part, please also cite the original repository and
publication listed in `Synthesizability-PU-CGCNN/README.md`:

- Jidon Jang, Geun Ho Gu, Juhwan Noh, Juhwan Kim, and Yousung Jung,
  "Structure-Based Synthesizability Prediction of Crystals Using Partially
  Supervised Learning", Journal of the American Chemical Society, 2020,
  142, 44, 18836-18843. DOI: https://doi.org/10.1021/jacs.0c07384

## Important Notes

- Scores from the two pipelines are model outputs for screening and ranking; they
  are not experimental proof that a material can or cannot be synthesized.
- `Synthesizability-PU-CGCNN/` is a copied and modified external codebase, kept
  here for the CLScore evaluation workflow.
- Large checkpoint files are included in the model directories. Keep the expected
  checkpoint names if you move or replace models.
