# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start Commands

### Testing
```bash
# Run all tests
pytest tests/

# Run single test file
pytest tests/test_pmv_ppd_iso.py

# Run specific test
pytest tests/test_pmv_ppd_iso.py::test_pmv_ppd

# Run tests matching pattern
pytest -k "pmv"

# Run with coverage
pytest tests/ --cov --cov-report=term-missing -vv

# Full test suite via tox (tests Python 3.10-3.13)
tox
```

### Linting and Formatting
```bash
# Check formatting
ruff format --check ./pythermalcomfort ./tests

# Apply formatting (in-place)
ruff format ./pythermalcomfort ./tests

# Lint check
ruff check ./pythermalcomfort ./tests

# Lint with auto-fix
ruff check --fix ./pythermalcomfort ./tests

# Format docstrings
docformatter -r -i --wrap-summaries 88 --wrap-descriptions 88 pythermalcomfort

# Run pre-commit hooks manually
pre-commit run --all-files
```

### CI/CD Workflow
- **Pull Request (development branch)**: Runs format checks and tests on Python 3.10, 3.13
- **Push to master**: Full test matrix (Python 3.10-3.13 on Ubuntu/macOS/Windows), then publishes to PyPI

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

**Class hierarchy in `plots/matplotlib/`:**
- `BasePlot` (`_base.py`): abstract base, defines `set_regions()` interface
- `GridBasePlot` (`_base.py`): extends BasePlot with model inspection, axis/param config, and grid evaluation — used by `ThresholdPlot` and `PsychrometricPlot`
- Standalone plots (e.g., `SummaryPlot`, `AdaptivePlot`) inherit directly from `BasePlot`

**Conventions:**
1. Use the fluent API pattern: setters return `self`, final `plot()` returns a `BasePlotResult` subclass
2. Define visual defaults as a nested class inside `_PlotDefaults` in `_shared.py`
3. Return Matplotlib handles (`fig`, `ax`, plus any extra artists) in the result dataclass for user customization
4. Export the new class from `plots/matplotlib/__init__.py`
5. Add a paired example notebook under `docs/documentation/plots/matplotlib/`

### Branch naming convention

- Features: `Feature/short-description`
- Bugfixes: `Fix/short-description`
- Documentation: `Documentation/doc-name`

### Pre-PR checklist

- Run `tox` or `pytest tests/` to verify all tests pass
- Run `ruff format` and `ruff check --fix` for code style
- Run `docformatter` for docstring formatting
- Add/update docstrings with NumPy style
- Update CHANGELOG.rst if applicable
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
