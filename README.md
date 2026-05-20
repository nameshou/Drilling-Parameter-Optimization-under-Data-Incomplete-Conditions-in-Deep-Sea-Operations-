# Drilling Parameter Optimization under Data-Incomplete Conditions in Deep-Sea Operations

A hybrid framework combining historical case retrieval and conditional generation for drilling-parameter optimization when operational data are incomplete.

## Overview

This repository implements three main components:

- `HNSW++-CHA.py` — a fuzzy retrieval module using an HNSW-based multi-branch index to find historical similar cases.
- `CDMsDDIM.py` — a conditional diffusion generator that completes missing drilling parameters under partially observed conditions.
- `topsis.py` — a multi-criteria evaluation module (AHP + entropy weighting + TOPSIS) with built-in example data and radar-plot visualization for quick testing.

All source code is provided as individual Python files (no compressed archives included).

## Dependencies

Recommended to use a virtual environment. Core Python packages:

- numpy
- pandas
- scikit-learn
- matplotlib
- openpyxl

Optional / for specific features:

- torch (for CDMsDDIM training / inference; choose the build appropriate for your CUDA/CPU environment)
- hnswlib (for running HNSW++-based fuzzy retrieval)

Example install commands:

pip install numpy pandas scikit-learn matplotlib openpyxl
pip install torch      # choose appropriate build for CUDA/CPU
pip install hnswlib    # if you want to run HNSW++ examples

## Quick start / Quick test

1. Clone the repository:

   git clone https://github.com/nameshou/Drilling-Parameter-Optimization-under-Data-Incomplete-Conditions-in-Deep-Sea-Operations-.git
   cd Drilling-Parameter-Optimization-under-Data-Incomplete-Conditions-in-Deep-Sea-Operations-

2. (Optional) Create and activate a virtual environment:

   python -m venv .venv
   # Linux / macOS
   source .venv/bin/activate
   # Windows (PowerShell)
   .\.venv\\Scripts\\Activate.ps1

3. Install dependencies (example):

   pip install numpy pandas scikit-learn matplotlib openpyxl

4. Run the quick test (built-in TOPSIS example):

   python quicktest.py

The `topsis.py` example runs without requiring any external data files; it prints AHP/entropy/TOPSIS results and saves radar charts (radar_group*.png) to the working directory.

## Running the HNSW++ (fuzzy retrieval) example

- `HNSW++-CHA.py` includes a main example that reads an Excel file. Edit the `excel_path` variable in the `if __name__ == "__main__":` section to point to your data file (or to a relative sample if you add one).
- Install dependency if needed:

  pip install hnswlib

- Run:

  python "HNSW++-CHA.py"

## Running the conditional diffusion generator example (CDMsDDIM.py)

- `CDMsDDIM.py` contains a runnable example that reads an Excel file for training and then generates completed schemes.
- Edit the `excel_path` in the `if __name__ == "__main__":` block to point to your data (relative paths recommended).
- For CPU-only usage, PyTorch without CUDA will be used automatically. For GPU usage, install a CUDA-enabled PyTorch build and ensure a GPU is available.
- Run:

  python CDMsDDIM.py

Note: Training the diffusion model can be computationally expensive. For quick flow validation you can reduce epochs in the script (for instance, set epochs=1) — this only serves to test the runtime flow, not model performance.

## Repository layout (main files)

- topsis.py
- HNSW++-CHA.py
- CDMsDDIM.py
- Transformer.py
- PGM-Index.py
- README.md
- quicktest.py
