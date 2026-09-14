"""JOS-3 transient thermophysiology: simulation vs. human-subject data.

Manuscript reference: "pythermalcomfort: An Open-Source Python Package for
Thermal Comfort, Heat Stress, and Cold Stress" (Building Simulation, Climate
Change and Urban Overheating special issue). This script produces the figure
for **Example 4**, which was promised in the response to Reviewer 2 Comment 6
(the newly added JOS-3 dynamic model was under-represented in the original
examples) and doubles as the external/observational comparison referenced in
the response to Reviewer 3 Comment 2.

What it demonstrates
---------------------
JOS-3 is a 17-segment, multi-node transient thermophysiology model, i.e. it
predicts how core and skin temperature *evolve over time* as the thermal
environment changes, not just a single steady-state comfort index. To show
that capability against real data, this script reproduces two step-change
exposures from the human-subject calorimetry experiments of Stolwijk and
Hardy (1966): a warm/hot transient and a cool/cold transient, each following
the same protocol used to validate JOS-3 in the package's own test suite
(``examples/calc_jos3.py::validation_simulation``):

1. A 100-min preconditioning period at 28 degC so every subject model starts
   from a comparable thermal state.
2. A 60-min thermoneutral baseline (~28 degC).
3. A 120-min step change to the challenge condition (hot or cold).
4. A 60-min recovery back to ~28 degC.

For each condition, three JOS-3 models are built with the anthropometry of
the three human subjects reported in Stolwijk and Hardy (1966) and their
predictions are averaged, mirroring how the experimental curves are
themselves averages across subjects. Simulated rectal (core) and mean skin
temperature are then plotted against the digitised experimental time series
supplied with the package, and the root-mean-square error (RMSE) between the
two is reported directly on the figure, so the plot functions as a
quantitative external validation, not just a qualitative demonstration.

Data provenance
----------------
J.A.J. Stolwijk, J.D. Hardy, "Partitional calorimetric studies of responses
of man to thermal transients", J. Appl. Physiol. 21(3) (1966) 967-977,
https://doi.org/10.1152/jappl.1966.21.3.967, and the companion paper J.D.
Hardy, J.A.J. Stolwijk, "Partitional calorimetric studies of man during
exposures to thermal transients", J. Appl. Physiol. 21(6) (1966) 1799-1806,
https://doi.org/10.1152/jappl.1966.21.6.1799 (both are commonly cited
together as "Stolwijk and Hardy, 1966"; see also Werner 1980 for the related
sheet shipped in the same dataset). The digitised experimental values are the
same Stolwijk1966 data already shipped and provenance-cleared for
``examples/calc_jos3.py``; this script reuses that file rather than
duplicating it. It is read from the CSV exported from the archival
spreadsheet, so running this example needs no Excel reader.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pythermalcomfort.models import JOS3

# The digitised Stolwijk & Hardy (1966) / Werner (1980) dataset already lives
# next to examples/calc_jos3.py; reuse it instead of shipping a second copy.
DATASET_PATH = (
    Path(__file__).resolve().parent.parent
    / "jos3_output_example"
    / "human_subject_experiment_dataset_Stolwijk1966.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_FIGURE = OUTPUT_DIR / "jos3_transient_validation.pdf"

# Local insulation pattern used throughout the Stolwijk & Hardy validation,
# matching examples/calc_jos3.py exactly: 0.3 clo over back, pelvis, thighs and
# legs, otherwise nude (order follows JOS3.body_names).
LOCAL_CLO = np.array(
    [0, 0, 0, 0.3, 0.3, 0, 0, 0, 0, 0, 0, 0.3, 0.3, 0, 0.3, 0.3, 0],
)

# Duration [min] of each of the four simulated phases: preconditioning,
# thermoneutral baseline, step-change exposure, recovery.
PHASE_MINUTES = (100, 60, 120, 60)


@dataclass(frozen=True)
class Subject:
    """Anthropometry of one Stolwijk & Hardy (1966) human subject."""

    height: float  # [m]
    weight: float  # [kg]
    age: int  # [years]


@dataclass(frozen=True)
class TransientCondition:
    """One step-change exposure reproduced from Stolwijk & Hardy (1966)."""

    label: str
    sheet_condition: str  # matches the "Condition" column in the dataset
    operative_temp: tuple[float, float, float, float]  # degC, per phase
    relative_humidity: tuple[float, float, float, float]  # %, per phase
    subjects: tuple[Subject, ...]


# Subjects and boundary conditions reproduced from Stolwijk & Hardy (1966),
# figures A-FIG.6 (hot transient) and B-FIG.2 (cold transient).
CONDITIONS = (
    TransientCondition(
        label="Hot transient (42.5°C)",
        sheet_condition="A-FIG.6",
        operative_temp=(28.0, 28.0, 42.5, 28.1),
        relative_humidity=(40, 37, 34, 37),
        subjects=(
            Subject(height=1.95, weight=88.6, age=25),
            Subject(height=1.84, weight=76.1, age=22),
            Subject(height=1.75, weight=110.0, age=23),
        ),
    ),
    TransientCondition(
        label="Cold transient (18°C)",
        sheet_condition="B-FIG.2",
        operative_temp=(28.0, 28.0, 18.0, 28.0),
        relative_humidity=(40, 40, 40, 40),
        subjects=(
            Subject(height=1.91, weight=77.2, age=25),
            Subject(height=1.91, weight=84.5, age=26),
            Subject(height=1.88, weight=92.7, age=22),
        ),
    ),
)


def simulate_condition(condition: TransientCondition) -> pd.DataFrame:
    """Simulate one four-phase exposure, averaged across its subjects.

    Each subject is simulated independently through the four phases and the
    resulting whole-body outputs are averaged, mirroring how the reference
    experimental curves are themselves subject-averages. Only the recorded
    portion (baseline + exposure + recovery, i.e. phases 2-4) is returned, at
    1-minute resolution, so the index runs from 0 to 240 min like the
    digitised experimental data.
    """
    precondition_min, baseline_min, exposure_min, recovery_min = PHASE_MINUTES
    to_phases = condition.operative_temp
    rh_phases = condition.relative_humidity

    per_subject_results = []
    for subject in condition.subjects:
        model = JOS3(height=subject.height, weight=subject.weight, age=subject.age)
        # NOTE: JOS3 exposes clothing insulation as .clo (backed by _clo).
        # Assigning .icl silently creates an unused attribute, leaving the
        # model nude (Default.clothing_insulation = 0).
        model.clo = LOCAL_CLO
        model.par = 1.2
        model.posture = "sitting"

        # Phase 1: coarse-step preconditioning so the model reaches a
        # comparable initial thermal state before the recorded phases start.
        model.to = to_phases[0]
        model.rh = rh_phases[0]
        model.simulate(precondition_min // 10, dtime=600)

        # Phases 2-4: 1-minute resolution baseline, exposure, and recovery.
        for to, rh, minutes in zip(
            to_phases[1:],
            rh_phases[1:],
            (baseline_min, exposure_min, recovery_min),
            strict=True,
        ):
            model.to = to
            model.rh = rh
            model.simulate(minutes)

        # ``dict_results()`` mis-zips its per-body-part columns onto the
        # wrong values in the installed release (each column ends up holding
        # its own body-part name instead of a temperature); ``results()``
        # builds the same time series correctly, so use it instead.
        output = model.results()
        # results() includes the initial state as row 0, so the last
        # preconditioning step (minute 0 of the recorded window) is at index
        # precondition_min // 10; everything after it is the baseline,
        # exposure, and recovery phases actually being compared to data.
        start = precondition_min // 10
        result = pd.DataFrame(
            {
                "t_skin_mean": output.t_skin_mean[start:],
                "t_core_pelvis": output.t_core.pelvis[start:],
            },
        )
        per_subject_results.append(result.reset_index(drop=True))

    averaged = sum(per_subject_results) / len(per_subject_results)
    total_minutes = baseline_min + exposure_min + recovery_min
    averaged.index = pd.RangeIndex(total_minutes + 1)
    return averaged


def load_reference_data(sheet_condition: str) -> pd.DataFrame:
    """Load the digitised experimental time series for one condition."""
    # float_precision="round_trip" keeps the parsed values bit-identical to the
    # archival spreadsheet these CSVs are exported from.
    reference = pd.read_csv(DATASET_PATH, float_precision="round_trip")
    reference = reference.loc[reference["Condition"] == sheet_condition].copy()
    reference.index = pd.RangeIndex(len(reference)) * 5  # data is 5-min spaced
    return reference


def rmse(simulated: pd.Series, reference: pd.Series) -> float:
    """Root-mean-square error between simulated and reference series."""
    aligned_sim = simulated.reindex(reference.index)
    return float(np.sqrt(np.mean((aligned_sim - reference) ** 2)))


def plot_condition(ax: plt.Axes, condition: TransientCondition) -> None:
    """Plot simulated vs. reference core and skin temperature on one axis."""
    simulated = simulate_condition(condition)
    reference = load_reference_data(condition.sheet_condition)

    core_rmse = rmse(simulated["t_core_pelvis"], reference["Tre"])
    skin_rmse = rmse(simulated["t_skin_mean"], reference["Tsk"])

    # Also report the errors on stdout, so the numbers quoted in the manuscript
    # can be checked without reading them off the figure.
    print(
        f"{condition.label}: rectal RMSE {core_rmse:.2f} degC, "
        f"skin RMSE {skin_rmse:.2f} degC, {len(reference)} reference points",
    )

    ax.plot(
        reference.index,
        reference["Tre"],
        "o",
        markersize=4,
        color="0.15",
        label="Rectal (measured)",
    )
    ax.plot(
        simulated.index,
        simulated["t_core_pelvis"],
        "-",
        linewidth=2,
        color="#1f77b4",
        label="Rectal (JOS-3)",
    )
    ax.plot(
        reference.index,
        reference["Tsk"],
        "s",
        markersize=4,
        color="0.6",
        label="Mean skin (measured)",
    )
    ax.plot(
        simulated.index,
        simulated["t_skin_mean"],
        "--",
        linewidth=2,
        color="#d62728",
        label="Mean skin (JOS-3)",
    )

    _, baseline_min, exposure_min, _ = PHASE_MINUTES
    exposure_start, exposure_end = baseline_min, baseline_min + exposure_min
    ax.axvspan(exposure_start, exposure_end, color="0.9", zorder=0)

    ax.set_xlim(0, sum(PHASE_MINUTES[1:]))
    ax.set_ylim(28, 41)
    ax.set_xlabel("Time [min]")
    ax.set_title(condition.label)
    ax.text(
        0.02,
        0.03,
        f"RMSE: rectal {core_rmse:.2f}°C, skin {skin_rmse:.2f}°C",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )


def build_figure(conditions: tuple[TransientCondition, ...]) -> plt.Figure:
    """Build the two-panel hot/cold transient validation figure."""
    fig, axes = plt.subplots(1, len(conditions), figsize=(9, 4), sharey=True)
    for ax, condition in zip(axes, conditions, strict=True):
        plot_condition(ax, condition)

    axes[0].set_ylabel("Body temperature [°C]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.06),
        fontsize=9,
    )
    fig.suptitle(
        "JOS-3 transient simulation vs. Stolwijk & Hardy (1966) human-subject data",
        y=1.14,
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    """Run the JOS-3 transient validation and save the manuscript figure."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure(CONDITIONS)
    fig.savefig(OUTPUT_FIGURE, bbox_inches="tight")
    print(f"Saved figure to {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
