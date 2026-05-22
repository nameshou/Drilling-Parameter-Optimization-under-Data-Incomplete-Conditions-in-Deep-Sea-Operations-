# Drilling Parameter Optimization under Data-Incomplete Conditions in Deep-Sea Operations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides a hybrid framework that combines historical case retrieval and conditional generation to support drilling-parameter optimization when operational data are incomplete or partially observed in deep-sea environments. It accompanies the manuscript:

> **A Hybrid Framework Combining Historical Case Retrieval and Conditional Generation for Drilling Parameter Optimization under Data Incomplete Conditions in Deep-Sea Operations**
>
> Linhao Wang, Xiaojun Chen, Qiwei Ren, Zelong Han
>
> Institute of Exploration Technology, Chinese Academy of Geological Sciences, Tianjin, China

## Authors

- **Linhao Wang** — Institute of Exploration Technology, Chinese Academy of Geological Sciences, Tianjin, China. ORCID: 0009-0003-3178-9403. E-mail: w1158128023@163.com
- **Xiaojun Chen** — Institute of Exploration Technology, Chinese Academy of Geological Sciences, Tianjin, China.
- **Qiwei Ren** — Institute of Exploration Technology, Chinese Academy of Geological Sciences, Tianjin, China.
- **Zelong Han** — Institute of Exploration Technology, Chinese Academy of Geological Sciences, Tianjin, China. (Corresponding author: 785674682@qq.com)

## Highlights

- Fuzzy retrieval of historical cases using an HNSW-based multi-branch index to find similar drilling cases under partial observation.
- Conditional diffusion model that completes missing drilling parameters conditioned on available measurements and retrieved historical cases.
- Multi-criteria decision and ranking pipeline (AHP + entropy weighting + TOPSIS) for evaluating candidate parameter sets.
- Lightweight example scripts to run quick tests without large datasets or heavy training.

## Repository structure (important files)

- **HNSW++-CHA.py** — Fuzzy retrieval module based on HNSW (hnswlib). Builds a multi-branch index for partial/masked queries and retrieves nearest historical cases.
- **CDMsDDIM.py** — Conditional diffusion-based model that imputes/completes missing drilling parameters under partially observed conditions. Uses PyTorch for model definition, training, and inference.
- **topsis.py** — Multi-criteria evaluation utilities combining AHP (Analytic Hierarchy Process), entropy-based weighting, and TOPSIS ranking. Includes a built-in example dataset and radar-plot visualization.
- **Transformer.py** — Transformer-based model for drilling time prediction with sliding window.
- **PGM-Index.py** — Learned-index (PGM-index) utility for efficient exact-match lookup.
- **quicktest.py** — Driver script that runs short end-to-end checks of all five modules. Use this to verify environment and dependencies.
- **requirements.txt** — Python package dependencies.
- **LICENSE** — MIT License.

## Data source

The dataset and related materials used in this project are available from the OGS research repository: <https://ricerca.ogs.it/handle/20.500.14083/44943>

## Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

Core packages: numpy, pandas, scikit-learn, matplotlib, openpyxl

Optional / for specific features:
- **torch (PyTorch)** — required for training or running the diffusion model in CDMsDDIM.py and Transformer.py
- **hnswlib** — required to run the HNSW++ retrieval examples
- **pympler** — used by PGM-Index.py for memory measurement

## Data format / expected inputs

Example scripts read from Excel (.xlsx) files or use synthetic data generated in-program. Expected layout is a tabular dataset where each row is a historical case and columns correspond to engineering/drilling parameters and measured responses:

- identifiers (case ID, timestamp)
- input / control parameters (e.g., weight on bit, rotary speed, pump rate)
- measured observations and responses (e.g., penetration rate, torque, viscosity metrics)
- NaN is used to indicate missing values

Tip: Inspect the `if __name__ == "__main__":` blocks in the example scripts to see how `excel_path` and column names are read.

## Quick start

1. Clone the repository:

   ```bash
   git clone https://github.com/nameshou/Drilling-Parameter-Optimization-under-Data-Incomplete-Conditions-in-Deep-Sea-Operations-.git
   cd Drilling-Parameter-Optimization-under-Data-Incomplete-Conditions-in-Deep-Sea-Operations-
   ```

2. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # Linux / macOS
   source .venv/bin/activate
   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the quick test (synthetic data, no external files required):

   ```bash
   python quicktest.py
   ```

5. Run individual modules with your own data by placing a `sample_data.xlsx` file in the repository root, then:

   ```bash
   python topsis.py
   python "HNSW++-CHA.py"
   python "PGM-Index.py"
   python CDMsDDIM.py
   python Transformer.py
   ```

## Notes and best practices

- When working with the diffusion model, start with a small number of epochs and a small dataset to ensure the training loop runs correctly before scaling up.
- Keep your Excel input columns consistent with the example scripts. If your column names differ, change the `read_excel` parsing code or preprocess your data to match expected names.
- HNSW-based fuzzy retrieval works best when you normalize or standardize feature vectors prior to building the index.


## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
