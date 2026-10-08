from dataclasses import dataclass, fields

import numpy as np
import pandas as pd
import pytest

from pythermalcomfort.classes_input import BaseInputs, WorkIntensity, choice_field
from pythermalcomfort.utilities import Postures, Sex

CHOICES = [
    (
        "position",
        ["sitting", "standing", "standing, forced convection"],
        Postures.sitting,
    ),
    ("posture", ["sitting", "standing", "crouching"], Postures.crouching),
    ("sex", ["male", "female"], Sex.female),
    ("work_intensity", ["heavy", "moderate", "light"], WorkIntensity.MODERATE),
]


def test_choice_field_default_and_metadata():
    @dataclass
    class Inputs(BaseInputs):
        custom_choice: str = choice_field(["first", "second"], default="second")

    declared = fields(Inputs)[-1]
    assert declared.default == "second"
    assert dict(declared.metadata) == {"allowed": ["first", "second"]}
    assert Inputs().custom_choice == "second"
    assert Inputs(custom_choice="FIRST").custom_choice == "FIRST"
    with pytest.raises(ValueError, match="custom_choice must be one of"):
        Inputs(custom_choice="invalid")


@pytest.mark.parametrize("name,allowed,enum_value", CHOICES)
def test_choice_fields_metadata_and_none(name, allowed, enum_value):
    declared = next(f for f in fields(BaseInputs) if f.name == name)
    assert declared.default is None
    assert dict(declared.metadata) == {"allowed": allowed}
    assert getattr(BaseInputs(), name) is None


@pytest.mark.parametrize("name,allowed,enum_value", CHOICES)
@pytest.mark.parametrize(
    "kind", ["scalar", "uppercase", "list", "array", "enum", "series"]
)
def test_choice_fields_preserve_valid_values(name, allowed, enum_value, kind):
    value = {
        "scalar": allowed[0],
        "uppercase": allowed[0].upper(),
        "list": allowed,
        "array": np.array(allowed),
        "enum": enum_value,
        "series": pd.Series(allowed),
    }[kind]
    if kind == "enum" and name == "work_intensity":
        # Preserve the existing rejection of this str/Enum subclass.
        with pytest.raises(ValueError, match="work_intensity must be one of"):
            BaseInputs(**{name: value})
        return
    result = getattr(BaseInputs(**{name: value}), name)
    if kind == "series":
        assert result == allowed
    else:
        assert result is value


@pytest.mark.parametrize("name,allowed,enum_value", CHOICES)
@pytest.mark.parametrize("kind", ["scalar", "list", "array"])
def test_choice_fields_reject_invalid_values(name, allowed, enum_value, kind):
    value = {
        "scalar": "invalid",
        "list": [allowed[0], "invalid"],
        "array": np.array([allowed[0], "invalid"]),
    }[kind]
    with pytest.raises(ValueError) as error:
        BaseInputs(**{name: value})
    assert str(error.value) == f"{name} must be one of {allowed!r}"
