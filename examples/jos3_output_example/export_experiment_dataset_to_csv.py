"""Export human_subject_experiment_dataset.xlsx to one CSV per sheet.

The JOS-3 examples read the digitised human-subject data as CSV so that running
them needs no Excel reader (``openpyxl`` is not a dependency of this package and
is not installed in the test environments). The spreadsheet remains in the
repository as the archival source of the data; these CSVs are exported from it
verbatim, with no rounding, reordering, or column renaming.

Run this only when the spreadsheet itself changes::

    python examples/jos3_output_example/export_experiment_dataset_to_csv.py

This script is the one place that still needs ``openpyxl`` (``pip install
openpyxl``). Values are written with ``repr()``, which is the shortest string
that round-trips to the same float64, so readers using
``pd.read_csv(..., float_precision="round_trip")`` get bit-identical values to
``pd.read_excel``.

Data provenance
---------------
J.A.J. Stolwijk, J.D. Hardy, "Partitional calorimetric studies of responses of
man to thermal transients", J. Appl. Physiol. 21(3) (1966) 967-977, and
R. Werner et al. (1980); see the manuscript and ``calc_jos3.py`` for the full
citations.
"""

from __future__ import annotations

import csv
from pathlib import Path

import openpyxl

SOURCE = Path(__file__).resolve().parent / "human_subject_experiment_dataset.xlsx"


def export() -> None:
    """Write one CSV per worksheet, next to the source spreadsheet."""
    workbook = openpyxl.load_workbook(SOURCE, data_only=True)

    for sheet_name in workbook.sheetnames:
        worksheet = workbook[sheet_name]
        destination = SOURCE.parent / f"{SOURCE.stem}_{sheet_name}.csv"

        with destination.open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            for row in worksheet.iter_rows(values_only=True):
                if all(cell is None for cell in row):
                    continue
                writer.writerow(
                    [
                        ""
                        if cell is None
                        else (repr(cell) if isinstance(cell, float) else cell)
                        for cell in row
                    ],
                )

        print(f"{sheet_name} -> {destination.name}")


if __name__ == "__main__":
    export()
