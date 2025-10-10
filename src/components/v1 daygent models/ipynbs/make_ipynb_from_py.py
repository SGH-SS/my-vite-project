"""
Convert a Python script with cell markers into a Jupyter Notebook.

Cell marker format supported:
  - Lines starting with "# %%" begin a new cell. The remainder of the
    line (after the marker) is included as the first line of the cell.

Usage:
  python make_ipynb_from_py.py fronttest_gb1d_reverse_test.py fronttest_gb1d_reverse_test.ipynb
"""

import sys
import json
from pathlib import Path


def py_to_notebook(py_path: Path, ipynb_path: Path) -> None:
    source = py_path.read_text(encoding='utf-8').splitlines()

    cells = []
    buffer = []

    def flush():
        if buffer:
            cells.append({
                "cell_type": "code",
                "metadata": {},
                "source": [line + "\n" for line in buffer],
                "outputs": [],
                "execution_count": None,
            })
            buffer.clear()

    for line in source:
        if line.lstrip().startswith('# %%'):
            # Start a new cell; flush the previous
            flush()
            # Keep the marker line, but convert to a comment description
            buffer.append(line)
        else:
            buffer.append(line)

    flush()

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": sys.version.split()[0]
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    ipynb_path.write_text(json.dumps(nb, indent=2), encoding='utf-8')
    print(f"✅ Wrote notebook: {ipynb_path}")


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python make_ipynb_from_py.py <input.py> [output.ipynb]")
        sys.exit(1)

    py_path = Path(sys.argv[1]).resolve()
    if len(sys.argv) == 3:
        ipynb_path = Path(sys.argv[2]).resolve()
    else:
        ipynb_path = py_path.with_suffix('.ipynb')

    if not py_path.exists():
        print(f"Input not found: {py_path}")
        sys.exit(1)

    py_to_notebook(py_path, ipynb_path)


if __name__ == '__main__':
    main()


