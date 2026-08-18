# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start Commands

### Python environment

This project uses **pipenv** (the env PyCharm is linked to, Python 3.12). Always run
Python tooling through `pipenv run` — never bare `pytest`/`python3`, never `uvx`, and
never create a new virtualenv. Doing so produces a second, divergent environment: the
repo already accumulated a stray 523 MB `.venv-docs/` that way, duplicating what
`pipenv run tox -e docs` already does.

```bash
pipenv install --dev    # one-time setup
pipenv run python -c "import sys; print(sys.executable)"   # confirm the env
```

### Testing
```bash
# Run all tests
pipenv run pytest tests/

# Run single test file
pipenv run pytest tests/test_pmv_ppd_iso.py

# Run specific test
pipenv run pytest tests/test_pmv_ppd_iso.py::test_pmv_ppd

# Run tests matching pattern
pipenv run pytest -k "pmv"

# Run with coverage
pipenv run pytest tests/ --cov --cov-report=term-missing -vv

# Full test suite via tox (tests Python 3.10-3.14)
pipenv run tox

# Build the docs (do NOT hand-roll a sphinx venv — this env already exists)
pipenv run tox -e docs
```

### Linting and Formatting
```bash
# Check formatting
pipenv run ruff format --check ./pythermalcomfort ./tests

# Apply formatting (in-place)
pipenv run ruff format ./pythermalcomfort ./tests

# Lint check
pipenv run ruff check ./pythermalcomfort ./tests

# Lint with auto-fix
pipenv run ruff check --fix ./pythermalcomfort ./tests

# Format docstrings
pipenv run docformatter -r -i --wrap-summaries 88 --wrap-descriptions 88 pythermalcomfort

# Run pre-commit hooks manually
pipenv run pre-commit run --all-files
```

### CI/CD Workflow
- **Pull Request (development branch)**: Runs format checks and tests on Python 3.10, 3.13, 3.14
- **Tag `v*rc*` on development**: Runs tests and deploys to TestPyPI
- **Tag `v*` (non-RC) on master**: Runs full test matrix and deploys to PyPI

### Release Process

`bump-my-version` auto-commits and auto-tags (`commit = true`, `tag = true` in
`.bumpversion.toml`), and correctly picks between the `Xrc{N}` and plain `X.Y.Z`
serialize formats on its own (via `pre_n`'s `optional_value`) — so in the normal case
you don't need to touch git yourself after running it, just push.

0. **Check the `validation-data-comfort-models` pin is current.** `tests/conftest.py`'s
   `unit_test_data_prefix` is pinned to a tag of that repo, not `main`. Compare it against
   [that repo's latest tag](https://github.com/FedericoTartarini/validation-data-comfort-models/tags)
   and its `CHANGELOG.md`; if behind, bump the pin and re-run the affected tests before
   proceeding. See `CONTRIBUTING.rst`'s "Keeping the validation-data-comfort-models pin
   current" for details — don't ship a release still pointed at a stale tag.

1. **Update ``CHANGELOG.rst``** with all changes since the last release, then commit.

2. **On `development`**, cut an RC (deploys to TestPyPI for verification). Pick
   whichever part reflects the change (`patch` / `minor` / `major`); it lands on
   `X.Y.Zrc1` automatically:
   ```bash
   pipenv run bump-my-version bump minor   # or patch / major
   git push origin development --tags
   ```
3. **Verify** the package on TestPyPI and confirm the GitHub Actions `deploy-testpypi` job passed.
4. **Found an issue and need another RC?** Just increment the RC number:
   ```bash
   pipenv run bump-my-version bump pre_n
   git push origin development --tags
   ```
5. **Merge `development` → `master`** via PR and confirm CI passes.
6. **Checkout master and pull**:
   ```bash
   git checkout master && git pull
   ```
7. **Finalize the release** (deploys to PyPI). This still needs an explicit target
   since dropping the RC suffix isn't a single-word part bump:
   ```bash
   pipenv run bump-my-version bump --new-version X.Y.Z
   git push origin master --tags
   ```
8. **Confirm** the `Test and publish pythermalcomfort` action succeeded and the package is live on PyPI.
9. **Sync master back into development** so the version-bump commit is not lost:
   ```bash
   git checkout development && git pull
   git merge origin/master --no-edit
   git push origin development
   ```

> **Note**: RC tags (`v*rc*`) must be created from `development` — the CI workflow enforces
> this with a `merge-base` check. Final release tags (`v*`) must be created from `master`.
> `bump-my-version` may fail to auto-commit/tag if a pre-commit hook (e.g. `ruff format`)
> modifies a file mid-commit. If that happens, stage the reformatted file manually
> (`git add <file>`), commit with the message `bump-my-version` printed
> (`Bump version: A.B.C → X.Y.Z`), and tag manually:
> `git tag vX.Y.Z -m "Bump version: A.B.C → X.Y.Z"`.

## Architecture Overview

### Codebase Organization

**Main package structure** (`pythermalcomfort/`):

1. **`models/`** - Thermal comfort calculation functions (39 modules)
   - Core models: `pmv_ppd_iso.py`, `pmv_ppd_ashrae.py`, `utci.py`, `adaptive_ashrae.py`, `adaptive_en.py`
   - Physiological models: `two_nodes_gagge.py`, `two_nodes_gagge_sleep.py`, `jos3.py`
   - Specialized indices: `heat_index_lu.py`, `wbgt.py`, `pet_steady.py`, `phs.py`, `set_tmp.py`
   - All functions support both scalar and array inputs, accept optional `units` parameter ('SI' or 'IP'), and return dataclass objects

2. **`plots/matplotlib/`** - Plotting utilities for visualization
   - `threshold.py`: `ThresholdPlot` class for threshold-region charts (e.g., comfort zones)
   - `summary.py`: `SummaryPlot` class for horizontal/vertical summaries from DataFrames
   - `adaptive.py`: `AdaptivePlot` for adaptive comfort visualization
   - `psychrometric.py`: `PsychrometricPlot` for psychrometric charts
   - `_shared.py`: Shared utilities, configuration dataclasses, and visual defaults
   - All plotting classes are fluent APIs that return Matplotlib handles for customization

3. **`classes_input.py`** - Input validation layer
   - `BaseInputs` dataclass with metadata-driven field validation
   - Model-specific subclasses (e.g., `PMVPPDInputs`, `UTCIInputs`, `GaggeTwoNodesInputs`)
   - Validation via `__post_init__()` checks types, allowed values, and normalizes units
   - Handles pandas Series conversion to lists transparently

4. **`classes_return.py`** - Output value objects
   - Frozen dataclasses for all model outputs (e.g., `PMVPPD`, `UTCI`, `AdaptiveASHRAE`)
   - `AutoStrMixin` provides aligned, multi-line `__str__()` with array summarization
   - Supports dict-like access via `__getitem__()` (e.g., `result['pmv']`)

5. **`utilities.py`** - Core utilities and enums
   - Enums: `Models`, `Units`, `Sex`, `Postures`
   - Psychrometric functions: `p_sat()`, `psy_ta_rh()`, `dew_point_tmp()`, `wet_bulb_tmp()`
   - Unit conversion: `units_converter()`
   - Physical constants and helper functions

6. **`shared_functions.py`** - Shared helper functions
   - `valid_range()`: Filters array values to valid ranges (sets out-of-range to NaN)
   - `mapping()`: Maps numeric arrays to categorical stress categories (using dict of bin edges)
   - `_finalize_scalar_or_array()`: Converts 0-d arrays to Python scalars while preserving NaN

7. **`jos3_functions/`** - JOS-3 physiological model submodules
   - `construction.py`: Body model initialization and validation
   - `thermoregulation.py`: Physiological response calculations
   - `matrix.py`: Node and segment indexing constants

### Design Patterns

**Function signatures** - All model functions follow this pattern:
```python
def model_name(
    tdb: float | list[float],           # input (scalar or array)
    tr: float | list[float],            # ...
    # ... additional parameters
    units: str = Units.SI.value,        # optional: 'SI' or 'IP'
    limit_inputs: bool = True,          # optional: enforce standard applicability limits
    round_output: bool = True,          # optional: round results
) -> OutputDataclass:                   # returns frozen dataclass
```

**Input validation flow**:
1. Function receives raw inputs
2. Instantiates input dataclass (e.g., `PMVPPDInputs(tdb=tdb, tr=tr, ...)`)
3. Dataclass `__post_init__()` validates types, values, units via metadata
4. Raises exceptions for invalid inputs
5. Function proceeds with validated, normalized inputs

**Array handling**:
- Functions convert inputs to numpy arrays via `np.asarray()`
- Use `np.where()` for conditional logic on arrays
- Use `np.isnan()` to filter/validate ranges
- Return scalar for scalar input, array for array input (via `_finalize_scalar_or_array()`)

**Numba optimization**:
- Performance-critical functions (e.g., UTCI) use `@vectorize` and `@jit` decorators
- Allows element-wise operations without explicit Python loops

### Key Module Interactions

```
models/*.py (thermal calculations)
  ↓ imports
classes_input.py (validates inputs)
  ↓ imports
utilities.py (enums, constants, unit conversion)
  ↓ imports
shared_functions.py (array filtering, mapping, finalization)
classes_return.py (output dataclasses)
  ↓ imports
plots/matplotlib/*.py (visualization of model outputs)
```

### Testing Architecture

**Test structure** (`tests/`):
- `conftest.py`: Pytest fixtures for fetching remote validation data
  - `get_test_url()`: Builds URLs to JSON test data from external repo
  - `retrieve_data()`: Fetches remote JSON via HTTP
  - `validate_result()`: Compares function output to reference with tolerance
  - `is_equal()`: Tolerant comparison for scalars and arrays
- 30+ test files with pattern `test_<model>.py`
- **Data-driven**: Each test retrieves reference data from https://github.com/FedericoTartarini/validation-data-comfort-models
  - Allows validation against academic reference implementations
  - Tests multiple inputs per model in a single JSON payload

## Docstring Style

All functions use **NumPy docstring format**:
- `Parameters` section: type hints and descriptions
- `Returns` section: type and description of output
- `Examples` section: executable code blocks
- `Notes` section: warnings and applicability limits
- Citations: References as `[AuthorYear]_` (sphinx bibliography)

Example from `pmv_ppd_iso()`:
```python
def pmv_ppd_iso(
    tdb: float | list[float],
    ...
) -> PMVPPD:
    """Calculate PMV and PPD per ISO 7730.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C] in [°F] if `units` = 'IP'
    ...

    Returns
    -------
    PMVPPD
        A dataclass containing pmv, ppd, tsv. Access via result.pmv

    Examples
    --------
    .. code-block:: python

        result = pmv_ppd_iso(tdb=25, tr=25, vr=0.1, rh=50, met=1.4, clo=0.5)
        print(result.pmv)  # 0.17
    """
```

## Development Notes

### When modifying models

1. **Maintain input/output contracts**: Model functions must accept scalar and array inputs, return dataclass with same attributes
2. **Use limit_inputs consistently**: If model has applicability limits, enforce via `valid_range()` and return NaN
3. **Update classes_input.py**: Add validation rules for new parameters in dataclass metadata
4. **Update classes_return.py**: Create/update output dataclass for return values
5. **Test with arrays**: Ensure model works with both single values and 1-D arrays
6. **Add docstring examples**: Include both scalar and array examples

### When adding new plots

1. Use the fluent API pattern from `plots/matplotlib/`: setters return `self`, final `plot()` returns figure/axis
2. Pass `thresholds`, `labels`, and `colors` directly to `set_regions` (no wrapper dataclass needed)
3. Return Matplotlib handles (`ax`, lines, patches) for user customization
4. Document via docstrings and examples in `plots/README.md`

### Branch naming convention

- Features: `Feature/short-description`
- Bugfixes: `Fix/short-description`
- Documentation: `Documentation/doc-name`

### Pre-PR checklist

- Run `tox` or `pytest tests/` to verify all tests pass
- Run `ruff format` and `ruff check --fix` for code style
- Run `docformatter` for docstring formatting
- Add/update docstrings with NumPy style
- **Update CHANGELOG.rst** with all user-facing changes (required before release)
- Ensure new functions have parameter and return type hints

## Key Files Reference

| File | Purpose |
|------|---------|
| `setup.py` | Package metadata, dependencies, entry points |
| `ruff.toml` | Linting and formatting rules (Black-compatible) |
| `.pre-commit-config.yaml` | Pre-commit hooks (ruff format/check) |
| `tox.ini` | Test environments for Python 3.10-3.13 |
| `.github/workflows/pull-request.yml` | CI for PRs to development branch |
| `.github/workflows/build-test-publish.yml` | Full CI + PyPI publish on master push |
| `pythermalcomfort/__init__.py` | Package version and public API |
| `pythermalcomfort/models/__init__.py` | Exports all model functions |
