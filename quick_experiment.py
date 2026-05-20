# Quick experiment: run the built-in TOPSIS example

This quick experiment runs the existing TOPSIS example in `topsis.py` to produce
ranking outputs and radar chart images. It uses a subprocess to execute the
script so it behaves the same as running `python topsis.py`.

Usage:

    python quick_experiment.py

Notes:
- This script assumes you have installed the repository dependencies (see
  README.md) and that `python` on your PATH points to the desired Python
  interpreter.
- The script simply runs the `topsis.py` example (which does not require
  external Excel files) and prints the script output to the console.
