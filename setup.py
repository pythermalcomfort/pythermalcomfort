from __future__ import annotations

import re
from pathlib import Path

from setuptools import find_packages, setup


def read(*names: str, encoding: str = "utf8") -> str:
    """Read and return the contents of a text file located by path segments
    relative to this file's directory.

    Parameters
    ----------
    *names : str
        Sequence of path components under the project root.
    encoding : str, default "utf8"
        File encoding to use when reading.

    Returns
    -------
    str
        The full text content of the file.

    """
    base_dir = Path(__file__).parent
    file_path = base_dir.joinpath(*names)
    return file_path.read_text(encoding=encoding)


setup(
    name="pythermalcomfort",
    version="4.6.0",
    license="MIT",
    description=(
        "pythermalcomfort is a comprehensive toolkit for calculating "
        "thermal comfort indices, heat/cold stress metrics, and thermophysiological responses. "
        "It supports multiple models, including PMV, PPD, adaptive comfort, SET, "
        "UTCI, Heat Index, Wind Chill Index, and Humidex. "
        "The package also includes thermophysiological models like the two-node (Gagge) and multinode (JOS-3) models "
        "to estimate physiological responses such as core temperature, skin temperature, and skin wettedness. "
    ),
    long_description="{}\n{}".format(
        re.compile("^.. start-badges.*^.. end-badges", re.MULTILINE | re.DOTALL).sub(
            "",
            read("README.rst"),
        ),
        re.sub(":[a-z]+:`~?(.*?)`", r"``\1``", read("CHANGELOG.rst")),
    ),
    long_description_content_type="text/x-rst",
    author="Federico Tartarini",
    author_email="cbecomforttool@gmail.com",
    url="https://github.com/pythermalcomfort/pythermalcomfort",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Utilities",
    ],
    project_urls={
        "Documentation": "https://pythermalcomfort.readthedocs.io/",
        "Changelog": "https://pythermalcomfort.readthedocs.io/en/latest/changelog.html",
        "Issue Tracker": (
            "https://github.com/pythermalcomfort/pythermalcomfort/issues"
        ),
    },
    keywords=[
        "thermal comfort",
        "pmv",
        "heat stress",
        "cold stress",
        "thermal sensation",
        "thermal physiology",
        "meteorology",
        "climate analysis",
        "discomfort",
        "comfort",
        "thermal environment",
        "built environment",
    ],
    python_requires=">=3.10.0",
    install_requires=[
        "scipy",
        "numba",
        "numpy>=1.21,<2.3",
        "setuptools>=78.1.1",
    ],
    extras_require={
        "dev": [
            "pytest",
            "sphinx",
            "ruff",
            "docformatter",
            "matplotlib>=3.8",
            "pandas",
            "seaborn>=0.13",
        ],
        "plots": ["matplotlib>=3.8", "pandas", "seaborn>=0.13"],
    },
    entry_points={
        "console_scripts": [
            "pythermalcomfort = pythermalcomfort.cli:main",
        ],
    },
)
