"""
quick_experiment.py (DEPRECATED)

This file has been deprecated and replaced by quicktest.py as the project's quick test/driver script.
Please run `python quicktest.py` instead.

The README has been updated to mark quicktest.py as the recommended quick test file and to include the data source link.
"""

import sys
import warnings

warnings.warn(
    "quick_experiment.py is deprecated. Use 'python quicktest.py' for quick tests. See README.md for details.",
    DeprecationWarning,
)

print("quick_experiment.py is deprecated. Run: python quicktest.py")
sys.exit(0)
