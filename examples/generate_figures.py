"""
Run all `NN_*.py` scripts in this folder in lexicographic order.

Each script writes a single self-contained PNG to `examples/figures/`.
To generate only one figure, run the corresponding script directly.

Usage (from the repository root):

    pip install -e .
    python examples/generate_figures.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    scripts = sorted(here.glob("[0-9][0-9]_*.py"))
    if not scripts:
        raise SystemExit("No NN_*.py script was found in examples/.")
    for script in scripts:
        print("---", script.name, "---")
        subprocess.run([sys.executable, str(script)], check=True)
    print("Completed:", here / "figures")


if __name__ == "__main__":
    main()
