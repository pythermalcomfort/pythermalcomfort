=================================
Adding a Thermal Comfort Function
=================================

Use this guide when contributing a new thermal comfort model or utility
function to ``pythermalcomfort``.

Quick checklist
===============

Before opening a PR, confirm:

- [ ] Implementation added under ``pythermalcomfort/models/<module_name>.py``
- [ ] Input dataclass created/updated with validation in ``__post_init__``
- [ ] Return dataclass added/updated in ``classes_return.py``
- [ ] NumPy-style docstring with units, examples, and applicability limits
- [ ] Tests added: scalars, arrays, broadcasting, invalid inputs
- [ ] ``autofunction`` entry added in the relevant ``docs/`` RST file
- [ ] CHANGELOG and AUTHORS updated (if applicable)
- [ ] All tests pass; formatting and linting applied

Step-by-step guide
==================

1. Pick the module location
---------------------------

* Domain model → ``pythermalcomfort/models/<module_name>.py``
* Environmental calculation → ``pythermalcomfort/environment/<module_name>.py``
* Psychrometric calculation → ``pythermalcomfort/psychrometrics/<module_name>.py``
* Clothing calculation → ``pythermalcomfort/clothing/<module_name>.py``
* Private implementation helper → ``pythermalcomfort/_internal/<module_name>.py``
* Truly generic helper, enum, or unit conversion → ``pythermalcomfort/utilities.py``

2. Implement the function
-------------------------

Keep it small, pure, and documented. Use NumPy for numeric operations.
Follow this skeleton:

.. code-block:: python

    # pythermalcomfort/models/my_model.py
    import numpy as np

    from pythermalcomfort.classes_input import MyModelInputs
    from pythermalcomfort.classes_return import MyModelResult

    def my_model(
        tdb: float | list[float],
        v: float | list[float],
        units: str = "SI",
        limit_inputs: bool = True,
    ) -> MyModelResult:
        """One-line summary.

        Parameters
        ----------
        tdb : float or list of float
            Dry-bulb air temperature, [°C] in [°F] if ``units`` = 'IP'.
        v : float or list of float
            Air speed, [m/s].
        units : {'SI', 'IP'}, optional
            Unit system.  Defaults to 'SI'.
        limit_inputs : bool, optional
            If ``True``, returns NaN for inputs outside applicability limits.

        Returns
        -------
        MyModelResult
            Dataclass with fields ``result`` and ``valid``.

        Examples
        --------
        .. code-block:: python

            from pythermalcomfort.models import my_model

            out = my_model(tdb=25, v=0.1)
            print(out.result)  # 42.0
        """
        MyModelInputs(tdb=tdb, v=v, units=units)

        tdb = np.asarray(tdb, dtype=float)
        v = np.asarray(v, dtype=float)

        result = ...  # computation

        return MyModelResult(result=result)

3. Create / update the input dataclass
---------------------------------------

Add input dataclasses to ``pythermalcomfort/classes_input.py``.
Put type checks and physical/applicability checks in ``__post_init__``.

.. code-block:: python

    @dataclass
    class MyModelInputs(BaseInputs):
        tdb: float | int | list | np.ndarray = None
        v: float | int | list | np.ndarray = None

        def __post_init__(self):
            super().__post_init__()
            validate_type(self.tdb, "tdb", (float, int, list, np.ndarray))
            validate_type(self.v, "v", (float, int, list, np.ndarray))
            if np.any(np.asarray(self.v) < 0):
                raise ValueError("v must be non-negative")

4. Create / update the return dataclass
-----------------------------------------

Add output dataclasses to ``pythermalcomfort/classes_return.py``.
Inherit from ``AutoStrMixin`` for a formatted ``__str__`` and ``__getitem__``
dict-style access.

.. code-block:: python

    @dataclass
    class MyModelResult(AutoStrMixin):
        result: float | np.ndarray

5. Export the function
-----------------------

Export each public calculation from the ``__init__.py`` of its selected package
(``models``, ``environment``, ``psychrometrics``, or ``clothing``). For example,
add a model to ``pythermalcomfort/models/__init__.py`` so it is accessible as
``from pythermalcomfort.models import my_model``. Truly generic utilities remain
in ``pythermalcomfort/utilities.py`` and do not need a package export. Private
helpers under ``pythermalcomfort/_internal`` must not be publicly exported.

6. Write tests
---------------

Add tests under ``tests/test_my_model.py``.  Cover:

* Scalar inputs (single values).
* Vectorised inputs (lists and NumPy arrays).
* Broadcasting behaviour and consistent output shapes.
* Invalid inputs (``TypeError`` and ``ValueError`` cases).
* Edge cases (zeros, very small/large inputs).

.. code-block:: python

    import numpy as np
    import pytest
    from pythermalcomfort.models import my_model

    def test_scalar():
        out = my_model(tdb=25, v=0.1)
        assert out.result == pytest.approx(42.0, abs=0.1)

    def test_array():
        out = my_model(tdb=[25, 26], v=0.1)
        assert len(out.result) == 2

    def test_invalid_v():
        with pytest.raises(ValueError):
            my_model(tdb=25, v=-1.0)

7. Documentation
-----------------

* Add a short executable example to the function docstring (scalar and array).
* Add an ``.. autofunction::`` entry in the relevant ``docs/documentation/``
  RST file (e.g. ``docs/documentation/models.rst``).
* For larger tutorials, add a notebook under ``docs/documentation/``.

8. CHANGELOG and AUTHORS
--------------------------

* Add a line to ``CHANGELOG.rst`` describing the new function.
* Optionally add yourself to ``AUTHORS.rst``.

9. Final checks and PR
-----------------------

.. code-block:: bash

    ruff format ./pythermalcomfort ./tests
    ruff check --fix ./pythermalcomfort ./tests
    docformatter -r -i --wrap-summaries 88 --wrap-descriptions 88 pythermalcomfort
    pytest tests/

PR description should include:

* What the function computes and why it is useful.
* Applicability limits and physical constraints.
* How it was tested and any notes on numeric stability.

Validation rules common to most functions
==========================================

* **Non-negativity** — validate physical quantities that must be ≥ 0.
* **Domain checks** — avoid logs/roots of non-positive numbers.
* **Shape/broadcasting** — verify array shapes are compatible when multiple
  array inputs are accepted.
* **Units** — document expected units; convert via ``units_converter`` when
  ``units='IP'`` is supported.
* **Error types** — use ``TypeError`` for wrong types, ``ValueError`` for
  invalid values.

Reference implementations
==========================

Use these existing files as style references:

* ``pythermalcomfort/models/pmv_ppd_iso.py`` — full model with input/output
  dataclasses, limit_inputs, unit conversion.
* ``pythermalcomfort/models/adaptive_ashrae.py`` — model that exposes
  module-level constants (``SLOPE``, ``INTERCEPT``) for use by other modules.
* ``pythermalcomfort/environment/``, ``pythermalcomfort/psychrometrics/``, and
  ``pythermalcomfort/clothing/`` — focused public calculation packages.
* ``pythermalcomfort/_internal/`` — private implementation helpers.
* ``pythermalcomfort/utilities.py`` — stable generic helpers, enums, and unit
  conversions only.
