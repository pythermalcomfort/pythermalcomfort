# manuscript-v4 examples

Reproducible example scripts for the *Building Simulation* manuscript
**"pythermalcomfort: An Open-Source Python Package for Thermal Comfort, Heat
Stress, and Cold Stress"** (pythermalcomfort v4.0.0, this tag). Each script is
self-contained: it reads its input data from `data/` (where needed) and
writes its output figure(s) to `output/` (created on first run).

| Script | Manuscript figure / section |
|---|---|
| `pmv_utci_example.py` | Software description: PMV/PPD (ISO 7730) and UTCI code listing |
| `plotting_api_example.py` | Software description: `ThresholdPlot` plotting API code listing |
| `example-1.py` | Illustrative example 1: PMV/UTCI/Heat Index T-RH comparison and PMV psychrometric chart |
| `example-epw-analysis.py` | Illustrative example 2: Beijing hourly UTCI analysis from an EPW climate file |
| `example-field-study.py` | Illustrative example 3: PMV, adaptive comfort, and thermal preference analysis of Cozie field-study data |

## Running

```bash
pipenv run python examples/manuscript-v4/example-1.py
pipenv run python examples/manuscript-v4/pmv_utci_example.py
pipenv run python examples/manuscript-v4/plotting_api_example.py
pipenv run python examples/manuscript-v4/example-epw-analysis.py
pipenv run python examples/manuscript-v4/example-field-study.py
```

`example-epw-analysis.py` optionally accepts a path to an alternative EPW
file as its first argument; it defaults to `data/beijing.epw`.
