# GitHub Copilot instructions for pythermalcomfort

Purpose
- Generate code consistent with the repository's conventions: dataclasses for inputs,
  numpy/pandas usage, centralized validation helpers, tests, docs, and PR expectations.
- Be concise, produce runnable code, and include minimal but sufficient comments.

Core principles
- Keep responses concise and focused.
- Prefer clarity and maintainability over cleverness.
- Ask one clarifying question if requirements are ambiguous.
- Follow repository patterns (dataclasses, BaseInputs metadata, validate_type,
  Units/Enums).

Quick repo conventions (summary)
- Use type hints and NumPy-style docstrings.
- Validate inputs in dataclass __post_init__ (use BaseInputs where relevant).
- Use numpy for numerical ops (np.log, np.asarray), pandas for tabular work.
- Raise TypeError for wrong types, ValueError for invalid values.
- Tests: pytest, cover scalar/array/broadcasting/invalid cases.
- Format/lint: ruff, docformatter; run tests before PR.

How to add a function (concise, actionable)
- Quick checklist (must complete before PR):
  - [ ] Implementation added under the appropriate focused package.
  - [ ] Input dataclass created/updated with validation in __post_init__.
  - [ ] NumPy-style docstring with units, applicability limits, and example(s).
  - [ ] Tests added (scalars, arrays, broadcasting, invalid inputs).
  - [ ] Documentation (autofunction entry) updated.
  - [ ] CHANGELOG / AUTHORS updated if applicable.
  - [ ] Formatting/linting applied and tests pass.

- Step-by-step guide:
  1. Choose location
     - Domain model → pythermalcomfort/models/<name>.py
     - Ambient or physical-environment calculation → pythermalcomfort/environment/<name>.py
     - Moist-air property calculation → pythermalcomfort/psychrometrics/<name>.py
     - Clothing-insulation calculation → pythermalcomfort/clothing/<name>.py
     - Private implementation helper → pythermalcomfort/_internal/<name>.py
     - Truly generic helper, enum, or unit conversion → pythermalcomfort/utilities.py

  2. Implement the function
     - Keep it small and single-purpose.
     - Use numpy operations for vectorized behavior.
     - Docstring: Args (with units), Returns, Raises, Examples, Applicability.

     Example:
     ```python
     def my_func(x: float | np.ndarray) -> float | np.ndarray:
         """Short description.

         Args:
             x: value in meters.

         Returns:
             Computed value.

         Raises:
             ValueError: if x is negative.

         Examples
         --------
         >>> my_func(1.0)
         2.0
         """
         x_arr = np.asarray(x)
         if np.any(x_arr < 0):
             raise ValueError("x must be non-negative")
         return x_arr * 2
     ```

  3. Add an input dataclass when appropriate
     - Add to pythermalcomfort/classes_input.py (or adjacent to function).
     - In __post_init__:
       - call super().__post_init__() if inheriting BaseInputs,
       - convert pandas Series to list/np.array,
       - validate types with validate_type(...),
       - normalize numeric inputs to np.asarray for vectorized code,
       - enforce physical constraints (non-negativity, z2>z0, z1>z0, z2>z1 where required),
       - check broadcasting compatibility (np.broadcast_shapes, np.atleast_1d).

  4. Tests
     - Add tests/tests_<function>.py following patterns:
       - Scalar correctness
       - Vectorized (list and np.array)
       - Broadcasting and shape checks
       - Invalid inputs raising TypeError/ValueError
     - Keep tests deterministic and small.

  5. Documentation & autodoc
     - Add an ``.. autofunction:: pythermalcomfort.<package>.<function>`` entry in the docs
       (docs/reference or the file that gathers API docs).
     - Ensure docstring examples are minimal and runnable.

  6. CHANGELOG / AUTHORS
     - Add a short changelog entry for public API changes.
     - Optionally add contributor to AUTHORS.rst.

  7. Format & test locally
     - Recommended commands:
       ```bash
       ruff check --fix
       ruff format
       docformatter -r -i --wrap-summaries 88 --wrap-descriptions 88 pythermalcomfort
       pytest -q
       ```
     - For full CI: use tox as configured.

  8. Open a PR
     - Title: concise, e.g., "feat: add my_func for X calculation"
     - PR body: what, why, tests added, applicability/limits, numeric stability notes.
     - Ensure CI passes.

Branch naming
- Feature branches: Feature/your-feature-name
- Bugfix branches: Fix/your-bug-name
- Docs: Documentation/doc-name

Validation & numerical guidance (common patterns)
- Use np.log for array support; avoid math.log when arrays are accepted.
- Validate domain before computing logs/roots: ensure arguments > 0.
- Use np.atleast_1d and np.broadcast_to for consistent shapes.
- For inputs that must be strictly positive or > another var, check with clear errors:
  - ValueError("z2 must be > z0") etc.

Testing expectations
- Use pytest.
- Cover scalar and array cases, broadcasting, and invalid inputs.
- Use numpy.testing or pytest.approx for numeric comparisons.

PR checklist (add to PR description)
- [ ] New tests added and passing.
- [ ] Docstring updated with examples and applicability limits.
- [ ] Documentation (autofunction) updated if public API changed.
- [ ] CHANGELOG updated (if applicable).
- [ ] Linting and formatting applied.
- [ ] All CI checks pass.

Common commands (copyable)
```bash
# quick tests and formatting
ruff check --fix
ruff format
docformatter -r -i --wrap-summaries 88 --wrap-descriptions 88 pythermalcomfort
pytest -q

# run full CI locally
tox
```

Assumptions
- validate_type, Units, Enums (Postures, WorkIntensity, Sex) exist in module scope.
- Raise TypeError for bad types, ValueError for invalid values.
- Ask if broadcasting semantics or pandas support are unclear.
