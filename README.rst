.. image:: https://github.com/pythermalcomfort/pythermalcomfort/raw/development/docs/images/pythermalcomfort-3-short.png
  :align: center
  :alt: pythermalcomfort logo

================
pythermalcomfort
================

``pythermalcomfort`` is a Python toolkit for computing thermal comfort indices,
heat/cold stress metrics, and thermophysiological responses.
Its implementations adhere to international standards and peer-reviewed research,
offering researchers, engineers, and building scientists reliable,
standards-compliant calculations without the burden of implementing them manually.

.. important::
   When ``pythermalcomfort`` informs published work, please cite it as:

   .. code-block:: text

      Tartarini, F., Schiavon, S., 2020.
      pythermalcomfort: A Python package for thermal comfort research.
      SoftwareX 12, 100578.
      https://doi.org/10.1016/j.softx.2020.100578

Key Features
============

- **Thermal Comfort Models** – PMV/PPD, adaptive comfort assessments,
  SET, and more bundled into a single API surface.
- **Heat & Cold Stress Indices** – UTCI, Heat Index, Wind Chill, Humidex, and
  other commonly-referenced metrics.
- **Thermophysiological Modeling** – two-node (Gagge) and multinode (JOS-3)
  models for estimating core/skin temperatures and skin wettedness.
- **Standards Compliance** – Calculations based on ASHRAE 55, ISO 7730,
  EN 16798, and supporting references.
- **Vectorized Inputs** – Accepts scalars, lists, or NumPy arrays; most
  functions broadcast across input arrays automatically.
- **Pythonic API** – Simple, documented entry points that plug into analysis
  workflows and pipelines.
- **Rich Documentation** – Tutorials, examples, and reference guides for each
  supported model and index.
- **Active Development** – Frequent releases, new features, and responsive
  issue resolution.
- **Open Source** – MIT licensed and developed transparently on GitHub.

Why Choose pythermalcomfort?
============================

- **Precision** – Accurate evaluations of comfort and stress that engineers can
  trust.
- **Efficiency** – Eliminates repetitive code so teams can focus on insights,
  not implementation details.
- **Versatility** – Useful in building science, HVAC design, biometeorology,
  sports science, and thermal physiology.
- **Evidence-Based Decisions** – Supports data-driven HVAC sizing, occupant
  comfort strategies, and performance benchmarking.

Installation
============

Install from PyPI:

.. code-block:: bash

   pip install pythermalcomfort

For alternative installation instructions, including development builds and
optional dependencies, see the
`official docs <https://pythermalcomfort.readthedocs.io/en/latest/installation.html>`_.

Requirements
============

- Python 3.10+
- NumPy, SciPy, Numba, setuptools (installed automatically)
- Optional: pandas, Matplotlib, or other plotting libraries for examples and visualizations

Quick Start
===========

A few lines are all you need to get started:

.. code-block:: python

   from pythermalcomfort.models import pmv_ppd_iso, utci

   # Calculate PMV and PPD using ISO 7730 standard
   result = pmv_ppd_iso(
       tdb=25,   # dry-bulb temperature in °C
       tr=25,    # mean radiant temperature in °C
       vr=0.1,   # relative air speed in m/s
       rh=50,    # relative humidity in %
       met=1.4,  # metabolic rate in met
       clo=0.5,  # clothing insulation in clo
       model="7730-2005",
   )
   print(f"PMV: {result.pmv}, PPD: {result.ppd}")

   # Calculate UTCI for heat stress assessment
   result = utci(tdb=30, tr=30, v=0.5, rh=50)
   print(result.utci)

   # Most functions also accept arrays for bulk calculations
   result = utci(tdb=[28, 30, 35], tr=[28, 30, 35], v=0.5, rh=50)
   print(result.utci)

For a full list of models and indices, see the
`API reference <https://pythermalcomfort.readthedocs.io/en/latest/>`_.

Support pythermalcomfort
========================

Maintaining an open-source scientific package takes time. You can help by:

- `Sponsoring on GitHub <https://github.com/sponsors/FedericoTartarini>`_
- Submitting code, docs, or tests via a pull request
- Reporting reproducible bugs or feature requests in the
  `issue tracker <https://github.com/pythermalcomfort/pythermalcomfort/issues>`_
- Assisting with testing, translations, or PR reviews
- Starring or sharing the project to raise awareness

Contributions
=============

We welcome all contributions. Please read the
`contributing guide <https://pythermalcomfort.readthedocs.io/en/latest/contributing/index.html>`_
before you start.

Quick checklist
---------------

* Open an issue when planning large changes to align on scope.
* Fork the repo and create a feature branch.
* Add or update tests for new behavior.
* Run linters/formatters and fix the reported issues.
* Update docs or the changelog when the public API changes.
* Submit clear, focused PRs with related issues linked.

Common commands
---------------

.. code-block:: bash

    # clone your fork and add upstream remote
    git clone git@github.com:your-username/pythermalcomfort.git
    cd pythermalcomfort
    git remote add upstream git@github.com:pythermalcomfort/pythermalcomfort.git
    git fetch upstream

    # create a branch and work on it
    git checkout -b Feature/awesome-feature
    tox  # run the full matrix (slow)
    tox -e py312  # run a single env
    pytest -k test_name_fragment

    # fix linting/formatting
    ruff check --fix
    ruff format
    docformatter --in-place --wrap-summaries 88 --wrap-descriptions 88 pythermalcomfort/*.py

    # commit and push
    git add .
    git commit -m "feat: short description of change"
    git push origin Feature/awesome-feature

Release process
---------------

Releases are tag-driven and published via GitHub Actions Trusted Publishing
(OIDC — no ``PYPI_API_TOKEN`` or ``TEST_PYPI_API_TOKEN`` secret is required).

The standard cycle is:

1. Develop and test a release candidate on ``development`` → TestPyPI.
2. Merge ``development`` → ``master`` via pull request.
3. Finalize the version on ``master`` → PyPI.

Step 1 — pre-release on ``development`` (TestPyPI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git checkout development
    git pull --ff-only
    git fetch --tags --prune

    # Start the RC cycle for the next patch release (e.g. 3.9.8 → 3.9.9rc1):
    bump-my-version bump patch

    # Push the bump commit and tag — this triggers tests + TestPyPI deploy:
    git push
    git push --tags

If the RC needs additional fixes, make the commits then create another RC:

.. code-block:: bash

    bump-my-version bump pre_n    # e.g. 3.9.9rc1 → 3.9.9rc2
    git push
    git push --tags

Step 2 — open and merge a pull request from ``development`` to ``master``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CI runs the full test suite on the PR. Once it passes, merge via GitHub.

Step 3 — finalize on ``master`` (PyPI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git checkout master
    git pull --ff-only
    git fetch --tags --prune

    # Strip the rc suffix to produce the stable version.
    # Replace X.Y.Z with the target version (e.g. 3.9.9):
    bump-my-version bump --new-version X.Y.Z patch

    # Push the bump commit and tag — this triggers tests + PyPI deploy:
    git push
    git push --tags

Rules:

* All RC tags (``vX.Y.ZrcN``) must be pushed from ``development``.
* All stable tags (``vX.Y.Z``) must be pushed from ``master`` after merging.
* Tag format: ``vX.Y.Z`` for stable, ``vX.Y.ZrcN`` for pre-release.
* Do not push a stable tag before the corresponding ``development`` → ``master``
  PR has been merged; the CI deploy job will reject it.

Getting Help
============

* Open an issue on GitHub with a minimal reproduction in the
  `issue tracker <https://github.com/pythermalcomfort/pythermalcomfort/issues>`_.
* Ask questions in PR comments for implementation guidance.
* Review the
  `contribution guidelines <https://pythermalcomfort.readthedocs.io/en/latest/contributing/index.html>`_
  for testing, documentation, and changelog expectations.
* Consult the API reference and examples at
  https://pythermalcomfort.readthedocs.io/en/latest/

Changelog
=========

A full list of changes per release is available in the
`CHANGELOG <https://github.com/pythermalcomfort/pythermalcomfort/blob/master/CHANGELOG.rst>`_.

License
=======

``pythermalcomfort`` is released under the MIT License.

Stats
=====

.. start-badges

.. list-table::
    :stub-columns: 1

    * - Documentation
      - |docs|
    * - License
      - |license|
    * - Downloads
      - |downloads|
    * - Tests
      - | |codecov|
        | |tests|
    * - Package
      - | |version| |wheel|
        | |supported-ver|
        | |package-health|

.. |tests| image:: https://github.com/pythermalcomfort/pythermalcomfort/actions/workflows/build-test-publish.yml/badge.svg
    :target: https://github.com/pythermalcomfort/pythermalcomfort/actions/workflows/build-test-publish.yml
    :alt: Tests to ensure pythermalcomfort works on different Python versions and OS

.. |package-health| image:: https://img.shields.io/badge/Snyk_security-monitored-8A2BE2
   :target: https://security.snyk.io/package/pip/pythermalcomfort
   :alt: Snyk Security Badge

.. |license| image:: https://img.shields.io/pypi/l/pythermalcomfort?color=brightgreen
    :target: https://github.com/pythermalcomfort/pythermalcomfort/blob/master/LICENSE
    :alt: pythermalcomfort license

.. |docs| image:: https://readthedocs.org/projects/pythermalcomfort/badge/?style=flat
    :target: https://readthedocs.org/projects/pythermalcomfort
    :alt: Documentation Status

.. |downloads| image:: https://img.shields.io/pypi/dm/pythermalcomfort?color=brightgreen
    :alt: PyPI - Downloads

.. |codecov| image:: https://codecov.io/github/pythermalcomfort/pythermalcomfort/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/pythermalcomfort/pythermalcomfort

.. |version| image:: https://img.shields.io/pypi/v/pythermalcomfort.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/pythermalcomfort

.. |wheel| image:: https://img.shields.io/pypi/wheel/pythermalcomfort.svg
    :alt: pythermalcomfort wheel
    :target: https://pypi.org/project/pythermalcomfort

.. |supported-ver| image:: https://img.shields.io/pypi/pyversions/pythermalcomfort.svg
    :alt: Supported versions
    :target: https://pypi.org/project/pythermalcomfort

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/pythermalcomfort.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/pythermalcomfort

.. end-badges
