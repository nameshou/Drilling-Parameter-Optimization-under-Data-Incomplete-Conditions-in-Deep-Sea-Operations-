# Drilling Parameter Optimization under Data-Incomplete Conditions in Deep-Sea Operations

This repository provides a hybrid framework that combines historical case retrieval and conditional generation to support drilling-parameter optimization when operational data are incomplete or partially observed. The code is intended for research and prototyping: it contains retrieval, generative, and multi-criteria evaluation modules along with small example drivers to demonstrate end-to-end flows.

## Highlights

- Fuzzy retrieval of historical cases using an HNSW-based multi-branch index to find similar drilling cases under partial observation.
- Conditional diffusion model that completes missing drilling parameters conditioned on available measurements and retrieved historical cases.
- Multi-criteria decision and ranking pipeline (AHP + entropy weighting + TOPSIS) for evaluating candidate parameter sets.
- Lightweight example scripts to run quick tests without large datasets or heavy training.

## Repository structure (important files)

- HNSW++-CHA.py
  - Fuzzy retrieval module based on HNSW (hnswlib). Builds a multi-branch index for partial/masked queries and retrieves nearest historical cases.
  - Example usage: edit the `excel_path` variable in the `if __name__ == "__main__"` block to point to your sample Excel file, then run the script.

- CDMsDDIM.py
  - Conditional diffusion-based model that imputes/completes missing drilling parameters under partially observed conditions.
  - Uses PyTorch for model definition, training, and inference. The file contains a runnable example that reads data from Excel, trains (or loads) a model, and generates completed schemes.
  - Notes: training can be expensive. For quick validation set epochs to a small value.

- topsis.py
  - Multi-criteria evaluation utilities combining AHP (Analytic Hierarchy Process), entropy-based weighting, and TOPSIS ranking.
  - Includes a built-in example dataset and a radar-plot visualization to test the pipeline without external data.

- Transformer.py
  - Transformer-based utilities and model components used as building blocks for conditional modeling (helper code used by CDMsDDIM or other experiments).

- PGM-Index.py
  - An experimental learned-index (PGM-index) utility used for efficient lookup or auxiliary indexing in some retrieval flows.

- quick_experiment.py and quicktest.py
  - Small driver scripts that run short end-to-end checks (TOPSIS quick test, example retrieval flows). Use these to verify environment and dependencies.

- README.md (this file)

## Dependencies

Recommended to use a Python virtual environment. Core packages used across scripts:

- numpy
- pandas
- scikit-learn
- matplotlib
- openpyxl

Optional / for specific features:

- torch (PyTorch) — required for training or running the diffusion model in CDMsDDIM.py
- hnswlib — required to run the HNSW++ retrieval examples

Install example:

pip install numpy pandas scikit-learn matplotlib openpyxl
# For model training / inference, install a matching PyTorch build for your platform (CPU or CUDA):
# See https://pytorch.org for the appropriate command
pip install torch
# If you want to run the HNSW++ examples
pip install hnswlib

## Data format / expected inputs

Most example scripts read from Excel (.xlsx) files. Expected layout is a tabular dataset where each row is a historical case and columns correspond to engineering/drilling parameters and measured responses. Typical columns used across scripts include (but are not limited to):

- identifiers (case ID, timestamp)
- input / control parameters (e.g., weight on bit, rotary speed, pump rate)
- measured observations and responses (e.g., penetration rate, torque, vibration metrics)
- mask or missing-value indicators (NaN is used to indicate missing values in examples)

Tip: Inspect the `if __name__ == "__main__":` blocks in the example scripts (HNSW++-CHA.py, CDMsDDIM.py, topsis.py) to see how `excel_path` and column names are read. Modify those variables to match the column names in your Excel file.

## Quick start

1. Clone the repository:

   git clone https://github.com/nameshou/Drilling-Parameter-Optimization-under-Data-Incomplete-Conditions-in-Deep-Sea-Operations-.git
   cd Drilling-Parameter-Optimization-under-Data-Incomplete-Conditions-in-Deep-Sea-Operations-

2. (Optional) Create and activate a virtual environment:

   python -m venv .venv
   # Linux / macOS
   source .venv/bin/activate
   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1

3. Install dependencies (example):

   pip install numpy pandas scikit-learn matplotlib openpyxl

4. Run the TOPSIS quick test (no external data required):

   python quicktest.py

5. Run HNSW++ example (if you have hnswlib and a sample Excel file):

   pip install hnswlib
   # edit HNSW++-CHA.py to set `excel_path` in the main block
   python "HNSW++-CHA.py"

6. Run conditional diffusion example (CDMsDDIM.py):

   # install pytorch appropriate for your platform
   python CDMsDDIM.py

## Notes and best practices

- When working with the diffusion model, start with a small number of epochs and a small dataset to ensure the training loop runs correctly before scaling up.
- Keep your Excel input columns consistent with the example scripts. If your column names differ, change the `read_excel` parsing code in the script or preprocess your data to match expected names.
- HNSW-based fuzzy retrieval works best when you normalize or standardize feature vectors prior to building the index.

