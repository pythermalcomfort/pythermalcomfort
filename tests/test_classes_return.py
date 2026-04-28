from dataclasses import dataclass

import pytest

# Import the AutoStrMixin class
from pythermalcomfort.classes_return import UTCI, AutoStrMixin


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


def test_autostr_includes_field_units() -> None:
    """Test that AutoStrMixin includes field units when defined."""
    result = UTCI(
        utci=[24.6, 40.6],
        stress_category=["no thermal stress", "very strong heat stress"],
    )

    output = str(result)

    assert "utci [C]" in output
    assert "stress_category" in output
    assert "stress_category [" not in output
