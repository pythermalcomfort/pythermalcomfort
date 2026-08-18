---
name: add-model
description: Add a new thermal comfort model to pythermalcomfort, wiring it across the model file, input/return dataclasses, package exports, tests, docs and CHANGELOG. Use when asked to add, implement, or port a new comfort model, index, or thermal metric.
---

# Adding a thermal comfort model

A model is not "added" until **seven** files agree. Miss one and the failure is
usually silent — the function imports fine but is absent from the public API, or
undocumented, or unvalidated. Work through the checklist in order.

Substitute throughout:
- `<name>` — snake_case function name, e.g. `esi`
- `<Name>` — the return dataclass, e.g. `ESI`

Use `esi` as the reference implementation; it is small and follows every current
convention.

## Checklist

| # | File | Change |
|---|---|---|
| 1 | `pythermalcomfort/classes_input.py` | `<Name>Inputs` class (+ any new `BaseInputs` field) |
| 2 | `pythermalcomfort/classes_return.py` | `<Name>` frozen dataclass |
| 3 | `pythermalcomfort/models/<name>.py` | the model function |
| 4 | `pythermalcomfort/models/__init__.py` | import **and** `__all__` |
| 5 | `tests/test_<name>.py` | tests |
| 6 | `docs/documentation/models.rst` | `autofunction` + `autoclass` |
| 7 | `docs/documentation/references.rst` | citation |
| 8 | `CHANGELOG.rst` | user-facing entry |

## 1. Input validation — `classes_input.py`

**Check `BaseInputs` first.** Every parameter must already be declared as a field
there. The field list is alphabetical. If your model takes a parameter that does not
exist yet (e.g. `wind_speed`), add it to `BaseInputs` in alphabetical position:

```python
    wind_speed: NumericInput = numeric_field()
```

This is the step most often missed, because omitting it fails at runtime rather than
at import.

Then add the model's input class. It passes only its own fields to `super()`,
leaving everything else `None`:

```python
class <Name>Inputs(BaseInputs):
    """Input class for <Model> calculation.

    This class validates and processes inputs required for calculating <Model>.
    """

    def __init__(self, tdb, rh, round_output=True):
        # Initialize with only required fields, setting others to None
        super().__init__(tdb=tdb, rh=rh, round_output=round_output)

    def __post_init__(self):
        super().__post_init__()

        rh = np.asarray(self.rh, dtype=float)
        if np.any(rh < 0) or np.any(rh > 100):
            raise ValueError("Relative humidity must be between 0 and 100 %")
```

Put model-specific range checks in `__post_init__`. Type checking is handled by
`BaseInputs` metadata — do not re-implement it.

## 2. Return value — `classes_return.py`

Frozen dataclass, `repr=False`, inheriting `AutoStrMixin`. The decorator is required;
without it `__str__` alignment and dict-style access break.

```python
@dataclass(frozen=True, repr=False)
class <Name>(AutoStrMixin):
    """Dataclass to represent the <Model>.

    Attributes
    ----------
    <name> : float or list of floats
        <Description>, [unit].
    """

    <name>: NumericInput
```

## 3. The model — `models/<name>.py`

```python
from __future__ import annotations

import numpy as np

from pythermalcomfort.classes_input import <Name>Inputs, NumericInput
from pythermalcomfort.classes_return import <Name>


def <name>(
    tdb: NumericInput,
    rh: NumericInput,
    round_output: bool = True,
) -> <Name>:
    """Calculate the <Model> [Author2001]_.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C].
    rh : float or list of floats
        Relative humidity, [%].
    round_output : bool, optional
        If True, rounds output value. If False, it does not round it. Defaults to True.

    Returns
    -------
    <Name>
        A dataclass containing the <Model>. See :py:class:`~pythermalcomfort.classes_return.<Name>` for more details.
        To access the `<name>` value, use the `<name>` attribute of the returned `<Name>` instance, e.g., `result.<name>`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import <name>

        result = <name>(tdb=30.2, rh=42.2)
        print(result.<name>)  # 26.2

        result = <name>(tdb=[30.2, 27.0], rh=[42.2, 68.8])
        print(result.<name>)  # [26.2, 25.6]
    """
    <Name>Inputs(tdb=tdb, rh=rh, round_output=round_output)

    tdb = np.asarray(tdb)
    rh = np.asarray(rh)

    _<name> = ...  # the model equation

    if round_output:
        _<name> = np.round(_<name>, 1)

    return <Name>(<name>=_<name>)
```

Notes on this shape:

- The `<Name>Inputs(...)` call is **constructed and discarded**. Validation happens in
  `__post_init__`; nothing consumes the instance. This looks like dead code — it is not.
- `np.asarray` on every numeric input keeps scalar and array paths identical.
- The `Examples` block must show **both** a scalar and a list call. Docs are built with
  the values inline as comments.
- The citation key (`[Author2001]_`) must exactly match the key added in step 7.
- If the model has applicability limits, take `limit_inputs: bool = True` and use
  `valid_range()` from `shared_functions.py` to set out-of-range entries to NaN.
- If it supports imperial units, take `units: str = Units.SI.value` and convert via
  `units_converter()`.

## 4. Export — `models/__init__.py`

**Two** edits, both alphabetical:

```python
from .<name> import <name>          # with the other imports
```
```python
    "<name>",                        # in __all__
```

Adding only the import is the easy mistake, and it does **not** fail loudly:
`from pythermalcomfort.models import <name>` still works, so your tests pass while
the model is absent from `__all__` — the documented public surface and what
`import *` exposes. The two lists agree exactly today; keep it that way (see the
parity check under Verify).

## 5. Tests — `tests/test_<name>.py`

Mirror `tests/test_esi.py`, which covers the five cases expected of every model:

1. scalar input against a known reference value
2. list input, returning a list
3. invalid numeric ranges → `pytest.raises(ValueError)`
4. boundary conditions (min/max of each input)
5. invalid types, e.g. strings → `pytest.raises(TypeError)`

Compare with `is_equal` from `tests.conftest`, never `==` — it tolerates float noise
and handles arrays:

```python
from tests.conftest import is_equal

def test_<name>() -> None:
    """Test that the function calculates <Model> correctly for given inputs."""
    result = <name>(tdb=30.2, rh=42.2)
    is_equal(result.<name>, 26.2, 0.1)
```

**Only if** validating against the remote fixture repo: add an entry to the `Urls`
enum in `tests/conftest.py` (`<NAME> = "ts_<name>.json"`) and note that the JSON must
exist in `validation-data-comfort-models` at the **pinned tag**, not just on its
`main`, or the test will 404.

## 6 & 7. Docs

`docs/documentation/models.rst` — add a section in the same style as its neighbours:

```rst
<Model Full Name> (<ABBR>)
--------------------------
.. autofunction:: pythermalcomfort.models.<name>.<name>

.. autoclass:: pythermalcomfort.classes_return.<Name>
    :members:
```

`docs/documentation/references.rst` — add the citation, key matching the docstring:

```rst
.. [Author2001] Author, A.B., Coauthor, C.D., 2001. Full title. Journal 26, 427-431. DOI: doi.org/10.xxxx/yyyy
```

## 8. CHANGELOG.rst

Add a user-facing entry. Required before release.

## Verify

Run all of these; each catches a different missed step.

```bash
# the function is actually exported
pipenv run python -c "from pythermalcomfort.models import <name>; print(<name>)"

# import list and __all__ agree - catches the step-4 half-edit, which no test catches
pipenv run python -c "
import re, pathlib, pythermalcomfort.models as m
src = pathlib.Path('pythermalcomfort/models/__init__.py').read_text()
names = {n.strip() for line in re.findall(r'^from \.\w+ import (.+)\$', src, re.M) for n in line.split(',')}
missing, extra = sorted(names - set(m.__all__)), sorted(set(m.__all__) - names)
print('imported but not in __all__:', missing)
print('in __all__ but not imported:', extra)
assert not missing and not extra, 'models/__init__.py is out of sync'
print('OK')"

# scalar and array paths both work
pipenv run python -c "
from pythermalcomfort.models import <name>
print(<name>(tdb=30.2, rh=42.2))
print(<name>(tdb=[30.2, 27.0], rh=[42.2, 68.8]))"

pipenv run pytest tests/test_<name>.py -v
pipenv run ruff format ./pythermalcomfort ./tests
pipenv run ruff check --fix ./pythermalcomfort ./tests
pipenv run docformatter -r -i --wrap-summaries 88 --wrap-descriptions 88 pythermalcomfort

# docs build, and the citation resolves - an unmatched key fails here, not earlier
pipenv run tox -e docs
```

A broken citation key surfaces only in the docs build, so do not skip it.
