from dataclasses import FrozenInstanceError, dataclass, is_dataclass

import pytest

from pythermalcomfort import classes_return

# Import the AutoStrMixin class
from pythermalcomfort.classes_return import AutoStrMixin
from pythermalcomfort.models import adaptive_en


@dataclass(repr=False)
class TestDataClass(AutoStrMixin):
    """A test dataclass to demonstrate the AutoStrMixin functionality."""

    field1: int
    field2: str
    field3: list


def test_autostr_with_dataclass() -> None:
    """Test that the AutoStrMixin generates a string representation for a dataclass."""
    obj = TestDataClass(field1=42, field2="test", field3=[1, 2, 3])
    expected_output = (
        "------------------\n  TestDataClass   \n------------------\n"
        "field1 : 42\nfield2 : test\nfield3 : [1, 2, 3]"
    )

    assert str(obj) == expected_output


def test_autostr_with_multiline_field() -> None:
    """Test that the AutoStrMixin handles multiline fields correctly."""
    obj = TestDataClass(field1=42, field2="test", field3=["line1", "line2"])
    expected_output = (
        "-----------------------\n     TestDataClass     \n"
        "-----------------------\nfield1 : 42\nfield2 : test\nfield3 : [line1, line2]"
    )

    assert str(obj) == expected_output


def test_autostr_empty_dataclass() -> None:
    """Test that the AutoStrMixin handles an empty dataclass correctly."""

    @dataclass
    class EmptyDataClass(AutoStrMixin):
        pass

    obj = EmptyDataClass()
    expected_output = "--------------\nEmptyDataClass\n--------------"
    assert str(obj) == expected_output


def test_autostr_repr_method() -> None:
    """Test that the __repr__ method returns the same as __str__."""
    obj = TestDataClass(field1=42, field2="test", field3=[1, 2, 3])
    expected_output = (
        "------------------\n  TestDataClass   \n------------------\n"
        "field1 : 42\nfield2 : test\nfield3 : [1, 2, 3]"
    )
    assert repr(obj) == expected_output
    # Verify that __repr__ returns the same as __str__
    assert repr(obj) == str(obj)


def test_autostr_getitem_method() -> None:
    """Test that the __getitem__ method works correctly."""
    obj = TestDataClass(field1=42, field2="test", field3=[1, 2, 3])
    assert obj["field1"] == 42
    assert obj["field2"] == "test"
    assert obj["field3"] == [1, 2, 3]


def test_autostr_getitem_method_key_error() -> None:
    """Test that __getitem__ raises AttributeError for non-existent keys."""
    obj = TestDataClass(field1=42, field2="test", field3=[1, 2, 3])
    with pytest.raises(KeyError):
        _ = obj["non_existent"]


RESULT_DATACLASSES = [
    obj
    for obj in vars(classes_return).values()
    if isinstance(obj, type)
    and is_dataclass(obj)
    and obj.__module__ == classes_return.__name__
]


@pytest.mark.parametrize("cls", RESULT_DATACLASSES, ids=lambda cls: cls.__name__)
def test_result_dataclasses_are_frozen(cls) -> None:
    """Every result dataclass is frozen."""
    assert cls.__dataclass_params__.frozen


def test_adaptive_en_result_is_frozen() -> None:
    """Assigning to an adaptive_en result raises, as for the other models."""
    result = adaptive_en(tdb=25, tr=25, t_running_mean=20, v=0.1)
    with pytest.raises(FrozenInstanceError):
        result.tmp_cmf = 0.0
    assert repr(result) == str(result)
