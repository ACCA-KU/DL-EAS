# HM-GAT

**Hierarchical Molecular Graph Attention Network for UV–Vis Absorption Spectrum Prediction**

HM-GAT is a graph-based deep-learning model for predicting UV–Vis absorption spectra of organic molecules in solution from molecular and solvent SMILES representations.

This repository provides the public inference package for the pretrained HM-GAT model, including the inference notebook, model definition and configuration, pretrained weights, runtime dependencies, and a public example dataset.

## Key features

- **SMILES-based input:** molecular and solvent structures are provided as SMILES strings.
- **Full-spectrum prediction:** predicts the absorption spectrum from **330 to 1000 nm** at **2 nm spacing**.
- **336-point spectral output:** each predicted spectrum contains **336 wavelength points**.
- **Solvent-aware prediction:** molecular and solvent representations are used jointly during inference.
- **Pretrained inference:** pretrained HM-GAT weights are included for direct spectrum prediction.
- **No DFT spectrum required:** inference does not require a DFT-calculated input spectrum.

## Repository structure

```text
HM_GAT/
├── README.md
├── LICENSE
├── HM_GAT_inference.ipynb
├── environment.yml
├── requirements.txt
├── .gitignore
│
├── model/
│   ├── HM_GAT_pretrained.pth
│   ├── config.yaml
│   └── network.py
│
├── data/
│   └── test_dataset.csv
│
├── resources/
│   └── functional_group.csv
│
└── D4CMPP2/
    └── ...
```

`D4CMPP2/` contains the subset of supporting modules required by the released HM-GAT inference workflow.

## Installation

### Recommended: Conda environment

For reproducibility, the recommended installation method is to create the environment directly from `environment.yml`.

```bash
conda env create -f environment.yml
conda activate hm_gat_reproduction
```

The environment uses Python 3.10 and installs DGL from the `dglteam/label/th24_cu118` Conda channel.

The validated environment uses the CUDA 11.8 builds of PyTorch and DGL.

### Optional: register the Conda environment as a Jupyter kernel

If your Jupyter frontend does not automatically detect the Conda environment, register it manually:

```bash
python -m ipykernel install --user --name hm_gat_reproduction --display-name "HM-GAT (Python 3)"
```

Then open `HM_GAT_inference.ipynb` with your preferred Jupyter frontend and select **HM-GAT (Python 3)** as the kernel.

### About `requirements.txt`

`requirements.txt` contains the pinned Pip-managed dependencies used by the release. DGL is intentionally not installed through `requirements.txt`; it is installed through Conda as specified in `environment.yml`.

Therefore, `environment.yml` is the recommended route for reproducing the validated environment.

## Quick start

1. Clone or download the repository.

2. Create and activate the validated environment:

```bash
conda env create -f environment.yml
conda activate hm_gat_reproduction
```

3. Open:

```text
HM_GAT_inference.ipynb
```

4. Run the notebook cells in order. The notebook includes a public example based on `data/test_dataset.csv`.

5. To predict another molecule–solvent pair, provide the corresponding molecular SMILES and solvent SMILES in the inference workflow.

## Input

HM-GAT uses two structural inputs:

- **Compound:** molecular SMILES
- **Solvent:** solvent SMILES

Only 2D SMILES-derived structural representations are required for the released inference workflow.

## Output

The inference workflow predicts a UV–Vis absorption spectrum over the following wavelength grid:

- **Range:** 330–1000 nm
- **Spacing:** 2 nm
- **Number of points:** 336

The prediction workflow generates wavelength-dependent predicted values using the columns:

- `Wavelength_nm`
- `Predicted`

The notebook also determines the wavelength corresponding to the maximum predicted absorption intensity.

## Public example dataset

The repository includes `data/test_dataset.csv`, containing **200 public test examples**.

The inference notebook directly uses the following core fields from this dataset:

- `compound` — molecular SMILES
- `solvent` — solvent SMILES
- `set` — dataset split label
- `330`, `332`, ..., `1000` — spectrum values on the 336-point wavelength grid

All released example rows are test-set examples. The public dataset is provided for demonstrating and reproducing the inference workflow.

## Pretrained model

The pretrained model files are located in:

```text
model/HM_GAT_pretrained.pth
model/config.yaml
model/network.py
```

`HM_GAT_pretrained.pth` contains the released pretrained HM-GAT model weights used by `HM_GAT_inference.ipynb`.

## Reproducibility

The released environment and notebook were validated in a fresh environment created from `environment.yml`.

The validated software stack includes:

- **Python:** 3.10
- **PyTorch:** 2.4.0+cu118
- **DGL:** 2.4.0+cu118
- **CUDA build:** 11.8

A clean-room execution of the released notebook completed successfully with:

- all substantive notebook cells executed
- zero notebook execution errors
- 200 public test examples available
- an exact 330–1000 nm / 2 nm wavelength grid
- a 336-point HM-GAT spectrum prediction

## Scope of this release

This repository is intended for inference and reproducibility with the released pretrained HM-GAT model.

The public package includes example data and the components required for inference; it is not intended to reproduce the complete original training dataset.

## Citation

Citation information for the corresponding HM-GAT publication will be added when the final publication metadata becomes available.

## License

This repository, including the bundled D4CMPP2 subset in this release, is distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

Copyright (c) 2026 ACCA-KU
