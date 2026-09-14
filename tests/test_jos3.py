from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pythermalcomfort.classes_return import JOS3BodyParts
from pythermalcomfort.jos3_functions import construction
from pythermalcomfort.jos3_functions.construction import (
    bfb_rate,
    capacity,
    conductance,
    local_bsa,
    validate_body_parameters,
    weight_rate,
)
from pythermalcomfort.jos3_functions.matrix import (
    IDICT,
    LAYER_NAMES,
    NUM_NODES,
    index_by_layer,
    index_order,
    local_arr,
    valid_index_by_layer,
    vessel_blood_flow,
)
from pythermalcomfort.jos3_functions.parameters import Default
from pythermalcomfort.jos3_functions.thermoregulation import (
    ava_blood_flow,
    basal_met,
    clo_area_factor,
    conv_coef,
    dry_r,
    error_signals,
    evaporation,
    fixed_hc,
    fixed_hr,
    local_mbase,
    local_q_work,
    nonshivering,
    operative_temp,
    rad_coef,
    resp_heat_loss,
    shivering,
    skin_blood_flow,
    sum_bf,
    wet_r,
)
from pythermalcomfort.models import JOS3


def test_jos3_class() -> None:
    """Test the JOS3 class and its methods."""
    # Test for the initialization of JOS3 class
    # Instantiate JOS3 class
    model = JOS3()
    # Check if the object is of type JOS3
    assert isinstance(model, JOS3)

    # Test for the methods of JOS3 class
    # Instantiate the JOS3 class
    model = JOS3()

    # Call the simulate method
    model.simulate(times=60)

    # Test the reset_setpt method
    result = model._reset_setpt()
    result = result.__dict__
    assert "t_core" in result
    assert "t_skin" in result
    # Check if the attributes of the JOS3 object have been updated as expected
    assert model.to is not None
    assert np.all(model.rh == 50)
    assert np.all(model.v == 0.1)
    assert np.all(model.clo == 0)
    assert model.par == 1.25
    # Check if the new set-point temperatures for core and skin have been set
    assert np.all(model.cr_set_point == model.t_core)
    assert np.all(model.sk_set_point == model.t_skin)

    # Test: JOS3 class with sex="female"
    model = JOS3(sex="female", bmr_equation="japanese")
    model.simulate(times=60)
    # Check if mean skin temperature is not NaN
    dict_output = model.dict_results()
    assert not np.isnan(dict_output["t_skin_mean"]).any()

    # Test the simulate method with different parameters
    model = JOS3(height=1.7, weight=60, age=30)
    # Set the first phase
    model.to = 28  # Operative temperature [°C]
    model.rh = 40  # Relative humidity [%]
    model.v = 0.2  # Air velocity [m/s]
    model.par = 1.2  # Physical activity ratio [-]
    model.simulate(60)  # Exposure time = 60 [min]

    # Set the next condition
    model.to = 20  # Change only operative temperature
    model.simulate(60)  # Additional exposure time = 60 [min]

    # Define output dictionary
    dict_output = model.dict_results()

    # Results to compare (example data in csv file)
    # Get the absolute path of the current script
    current_script_path = Path(__file__).resolve()
    # Get the project directory by going up two levels from the current script path
    project_directory = Path(current_script_path).parent.parent
    # Specify the relative path to the CSV file
    relative_path = (
        Path("examples") / "jos3_output_example" / "jos3_example1 (default output).csv"
    )
    # Generate the absolute path by combining the project and the relative path
    file_path = project_directory / Path(relative_path)

    # Read the 't_skin_mean' column from the CSV file into a DataFrame
    df_t_skin_mean = (
        pd.read_csv(file_path, skiprows=2)["mean skin temperature"]
    ).tolist()

    # Check if the simulate method returns the expected type (e.g., dict)
    assert isinstance(dict_output, dict)
    assert np.allclose(dict_output["t_skin_mean"], df_t_skin_mean)

    # Call the _run method
    model._run(dtime=60000, passive=True)
    result = model.dict_results()
    # Check if the _run method returns the expected type (e.g., dict)
    assert isinstance(result, dict)

    # Test: Check property
    # Instantiate the JOS3 class
    model = JOS3()
    # Access the t_core property
    t_core = model.t_core
    # Check if t_core is of the expected type (e.g., numpy.ndarray)
    assert isinstance(t_core, np.ndarray)

    # Test with value out of range
    with pytest.raises(ValueError):
        JOS3(weight=1)

    # Test with value out of range
    with pytest.raises(ValueError):
        JOS3(age=1)

    # Test with value out of range
    with pytest.raises(ValueError):
        JOS3(height=0.1)

    # Test with value out of range
    with pytest.raises(ValueError):
        JOS3(fat=91)


def test_body_parameters() -> None:
    """Test the validate_body_parameters function."""
    # Test with valid parameters
    validate_body_parameters(height=1.75, weight=70.0, age=30, body_fat=15)
    # Test with invalid height
    with pytest.raises(ValueError):
        validate_body_parameters(height=0.1, weight=70.0, age=30, body_fat=15)
    # Test with invalid weight
    with pytest.raises(ValueError):
        validate_body_parameters(height=1.75, weight=210.0, age=30, body_fat=15)
    # Test with invalid age
    with pytest.raises(ValueError):
        validate_body_parameters(height=1.75, weight=210.0, age=101, body_fat=15)
    # Test with invalid body fat
    with pytest.raises(ValueError):
        validate_body_parameters(height=1.75, weight=210.0, age=101, body_fat=91)


def test_to17array() -> None:
    """Test the to_array_body_parts function from construction.py."""
    # Test with integer input
    result = construction.to_array_body_parts(5)
    assert isinstance(result, np.ndarray)
    assert result.shape == (17,)
    assert np.all(result == 5)

    # Test with float input
    result = construction.to_array_body_parts(5.5)
    assert isinstance(result, np.ndarray)
    assert result.shape == (17,)
    assert np.all(result == 5.5)

    # Test with list input
    result: np.ndarray = construction.to_array_body_parts(list(range(17)))
    assert isinstance(result, np.ndarray)
    assert result.shape == (17,)
    assert np.all(result == np.arange(17))

    # Test with ndarray input
    result = construction.to_array_body_parts(np.arange(17))
    assert isinstance(result, np.ndarray)
    assert result.shape == (17,)
    assert np.all(result == np.arange(17))

    # Test with dict input
    dict_input = {name: i for i, name in enumerate(JOS3BodyParts.get_attribute_names())}
    result = construction.to_array_body_parts(dict_input)
    assert isinstance(result, np.ndarray)
    assert result.shape == (17,)
    assert np.all(result == np.arange(17))

    # Test with list input of wrong length
    with pytest.raises(ValueError):
        construction.to_array_body_parts(list(range(16)))

    # Test with ndarray input of wrong length
    with pytest.raises(ValueError):
        construction.to_array_body_parts(np.arange(16))

    # Test with unsupported input type
    with pytest.raises(ValueError):
        construction.to_array_body_parts("unsupported")


def test_bsa_rate() -> None:
    """Test the bsa_rate function from construction.py."""
    # Test with default parameters
    expected_result = (
        1.0  # Since height and weight are set to default values, bsa_rate should be 1.0
    )
    result = construction.bsa_rate(height=1.72, weight=74.43, bsa_equation="dubois")
    assert result == pytest.approx(
        expected_result,
        rel=1e-3,
    )  # a relative tolerance of 1e-3

    # Test with custom parameters
    result = construction.bsa_rate(height=1.8, weight=80, bsa_equation="dubois")
    assert isinstance(result, float)

    # Test with invalid formula
    with pytest.raises(ValueError):
        construction.bsa_rate(bsa_equation="sushi", height=1.72, weight=74.43)

    # Test with non-numeric height
    with pytest.raises(TypeError):
        construction.bsa_rate(height="non-numeric", weight=74.43, bsa_equation="dubois")

    # Test with non-numeric weight
    with pytest.raises(TypeError):
        construction.bsa_rate(weight="non-numeric", height=1.72, bsa_equation="dubois")


def test_local_bsa() -> None:
    """Test the local_bsa function from construction.py."""
    # Test with default parameters
    result = local_bsa(height=1.72, weight=74.43, bsa_equation="dubois")
    assert isinstance(result, np.ndarray)
    assert result.shape == (17,)

    # Test with custom parameters
    result = local_bsa(height=1.8, weight=80, bsa_equation="dubois")
    assert isinstance(result, np.ndarray)
    assert result.shape == (17,)

    # Test with invalid formula
    with pytest.raises(ValueError):
        local_bsa(bsa_equation="unknown", height=1.72, weight=74.43)

    # Test with non-numeric height
    with pytest.raises(TypeError):
        local_bsa(height="non-numeric", weight=74.43, bsa_equation="dubois")

    # Test with non-numeric weight
    with pytest.raises(TypeError):
        local_bsa(weight="non-numeric", height=1.72, bsa_equation="dubois")


def test_weight_rate() -> None:
    """Test the weight_rate function from construction.py."""
    # Test with default parameters
    result = weight_rate(weight=74.43)
    expected_result = (
        1.0  # Since height and weight are set to default values, bsa_rate should be 1.0
    )
    assert result == pytest.approx(
        expected_result,
        rel=1e-3,
    )  # a relative tolerance of 1e-3

    # Test with custom weight
    weight = 80.0  # A valid weight value in kg
    result = weight_rate(weight=weight)
    standard_weight = 74.43
    assert (
        result == weight / standard_weight
    )  # The result should be the input weight divided by the standard weight

    # Test with non-numeric weight
    with pytest.raises(TypeError):
        weight_rate(weight="non-numeric")


def test_bfb_rate() -> None:
    """Test the bfb_rate function from construction.py."""
    # Test with default parameters
    result = bfb_rate(height=1.72, weight=74.43, bsa_equation="dubois", age=20, ci=2.59)
    assert isinstance(result, float)

    # Test with custom parameters
    result = bfb_rate(height=1.8, weight=80, bsa_equation="dubois", age=30, ci=2.7)
    assert isinstance(result, float)

    # Test with different ages
    result_young = bfb_rate(
        age=40,
        height=1.72,
        weight=74.43,
        bsa_equation="dubois",
        ci=2.59,
    )
    result_old = bfb_rate(
        age=60,
        height=1.72,
        weight=74.43,
        bsa_equation="dubois",
        ci=2.59,
    )
    assert result_old < result_young  # The BFB rate should decrease with age

    # Test with invalid equation
    with pytest.raises(ValueError):
        bfb_rate(bsa_equation="unknown", height=1.72, weight=74.43, age=20, ci=2.59)

    # Test with non-numeric height
    with pytest.raises(TypeError):
        bfb_rate(
            height="non-numeric",
            weight=74.43,
            age=20,
            ci=2.59,
            bsa_equation="dubois",
        )

    # Test with non-numeric weight
    with pytest.raises(TypeError):
        bfb_rate(
            weight="non-numeric",
            height=1.72,
            age=20,
            ci=2.59,
            bsa_equation="dubois",
        )

    # Test with non-numeric age
    with pytest.raises(TypeError):
        bfb_rate(
            age="non-numeric",
            height=1.72,
            weight=74.43,
            ci=2.59,
            bsa_equation="dubois",
        )

    # Test with non-numeric ci
    with pytest.raises(TypeError):
        bfb_rate(
            ci="non-numeric",
            height=1.72,
            weight=74.43,
            age=20,
            bsa_equation="dubois",
        )


def test_conductance() -> None:
    """Test the conductance function from construction.py."""
    # Test with default parameters
    result = conductance(height=1.72, weight=74.43, bsa_equation="dubois", fat=15.0)
    assert isinstance(result, np.ndarray)

    # Test with custom parameters
    result = conductance(height=1.8, weight=80, bsa_equation="dubois", fat=20.0)
    assert isinstance(result, np.ndarray)

    # Test with different fat rates
    result_low_fat = conductance(
        height=1.72,
        weight=74.43,
        bsa_equation="dubois",
        fat=10.0,
    )
    result_high_fat = conductance(
        height=1.72,
        weight=74.43,
        bsa_equation="dubois",
        fat=30.0,
    )
    assert np.any(
        result_low_fat != result_high_fat,
    )  # The conductance matrix should differ

    # Test with non-numeric height
    with pytest.raises(TypeError):
        conductance(height="non-numeric", weight=74.43, bsa_equation="dubois", fat=10.0)

    # Test with non-numeric weight
    with pytest.raises(TypeError):
        conductance(height=1.72, weight="non-numeric", bsa_equation="dubois", fat=10.0)

    # Test with non-numeric fat
    with pytest.raises(TypeError):
        conductance(height=1.72, weight=74.43, fat="non-numeric", bsa_equation="dubois")


def test_capacity() -> None:
    """Test the capacity function from construction.py."""
    # Test with default parameters
    result = capacity(height=1.72, weight=74.43, bsa_equation="dubois", age=20, ci=2.59)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == NUM_NODES

    # Test with non-default parameters
    result = capacity(height=1.8, weight=80, bsa_equation="takahira", age=25, ci=3)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == NUM_NODES

    # Test with invalid equation
    with pytest.raises(ValueError):
        capacity(
            bsa_equation="invalid_equation",
            height=1.72,
            weight=74.43,
            age=20,
            ci=2.59,
        )


def test_index_order() -> None:
    """Test the index_order function from matrix.py."""
    index_dict, order_count = index_order()

    # Test that output is of correct type
    assert isinstance(index_dict, dict)
    assert isinstance(order_count, int)

    # Test that output has correct keys and structure
    for key in JOS3BodyParts.get_attribute_names():
        assert key in index_dict
        for sub_key in LAYER_NAMES:
            assert sub_key in index_dict[key]

    # Test that "CB" key is present and it's value is 0
    assert "CB" in index_dict
    assert index_dict["CB"] == 0

    # Test order_count is correct
    total_layers = sum(
        1
        for bn in JOS3BodyParts.get_attribute_names()
        for ln in LAYER_NAMES
        if index_dict[bn][ln] is not None
    )
    assert total_layers + 1 == order_count  # +1 because of the "CB" key


def test_index_by_layer() -> None:
    """Test the index_by_layer function from matrix.py."""
    # Test that output is of correct type and length for each layer
    for layer in LAYER_NAMES:
        indices = index_by_layer(layer)
        assert isinstance(indices, list)
        assert len(indices) == sum(
            1
            for bn in JOS3BodyParts.get_attribute_names()
            if IDICT[bn][layer] is not None
        )

    # Test that each index is in the correct range
    for layer in LAYER_NAMES:
        indices = index_by_layer(layer)
        for index in indices:
            assert (
                0
                <= index
                < sum(
                    1
                    for bn in JOS3BodyParts.get_attribute_names()
                    for ln in LAYER_NAMES
                    if IDICT[bn][ln] is not None
                )
                + 1
            )  # +1 because of the "CB" key


def test_valid_index_by_layer() -> None:
    """Test the valid_index_by_layer function from matrix.py."""
    # Test that output is of correct type and length for each layer
    for layer in LAYER_NAMES:
        indices = valid_index_by_layer(layer)
        assert isinstance(indices, list)
        assert len(indices) == sum(
            1
            for bn in JOS3BodyParts.get_attribute_names()
            if IDICT[bn][layer] is not None
        )

    # Test that each index is in the correct range
    for layer in LAYER_NAMES:
        indices = valid_index_by_layer(layer)
        for index in indices:
            assert 0 <= index < len(JOS3BodyParts.get_attribute_names())

    # Test for incorrect layer input
    with pytest.raises(KeyError):
        valid_index_by_layer("non_existent_layer")


def test_local_arr() -> None:
    """Test the local_arr function from matrix.py."""
    # Initialize some random blood flow values
    rng = np.random.default_rng(1337)
    bf_core = rng.random(len(JOS3BodyParts.get_attribute_names()))
    bf_muscle = rng.random(len(JOS3BodyParts.get_attribute_names()))
    bf_fat = rng.random(len(JOS3BodyParts.get_attribute_names()))
    bf_skin = rng.random(len(JOS3BodyParts.get_attribute_names()))
    bf_ava_hand = rng.random(1)[0]
    bf_ava_foot = rng.random(1)[0]

    # Call the function with these values
    result = local_arr(bf_core, bf_muscle, bf_fat, bf_skin, bf_ava_hand, bf_ava_foot)

    # Check that the result is a 2D array with the correct dimensions
    assert isinstance(result, np.ndarray)
    assert result.shape == (NUM_NODES, NUM_NODES)

    # Check that the elements in the array are correct
    # You could add more specific checks here, depending on the expected
    # behavior of the function
    for i, bn in enumerate(JOS3BodyParts.get_attribute_names()):
        index_of = IDICT[bn]

        assert np.isclose(
            result[index_of["core"], index_of["artery"]],
            1.067 * bf_core[i],
        )
        assert np.isclose(
            result[index_of["skin"], index_of["artery"]],
            1.067 * bf_skin[i],
        )
        assert np.isclose(
            result[index_of["vein"], index_of["core"]],
            1.067 * bf_core[i],
        )
        assert np.isclose(
            result[index_of["vein"], index_of["skin"]],
            1.067 * bf_skin[i],
        )

        if index_of["muscle"] is not None:
            assert np.isclose(
                result[index_of["muscle"], index_of["artery"]],
                1.067 * bf_muscle[i],
            )
            assert np.isclose(
                result[index_of["vein"], index_of["muscle"]],
                1.067 * bf_muscle[i],
            )
        if index_of["fat"] is not None:
            assert np.isclose(
                result[index_of["fat"], index_of["artery"]],
                1.067 * bf_fat[i],
            )
            assert np.isclose(
                result[index_of["vein"], index_of["fat"]],
                1.067 * bf_fat[i],
            )

        if i in [7, 10]:
            assert np.isclose(
                result[index_of["sfvein"], index_of["artery"]],
                1.067 * bf_ava_hand,
            )
        if i in [13, 16]:
            assert np.isclose(
                result[index_of["sfvein"], index_of["artery"]],
                1.067 * bf_ava_foot,
            )


def test_vessel_blood_flow() -> None:
    """Test the vessel_blood_flow function from matrix.py."""
    # Initialize some random blood flow values
    rng = np.random.default_rng(1337)
    bf_core = rng.random(17)
    bf_muscle = rng.random(17)
    bf_fat = rng.random(17)
    bf_skin = rng.random(17)
    bf_ava_hand = rng.random(1)[0]
    bf_ava_foot = rng.random(1)[0]

    # Call the function with these values
    bf_art, bf_vein = vessel_blood_flow(
        bf_core,
        bf_muscle,
        bf_fat,
        bf_skin,
        bf_ava_hand,
        bf_ava_foot,
    )

    # Check that the result is a 1D array with the correct dimensions
    assert isinstance(bf_art, np.ndarray)
    assert isinstance(bf_vein, np.ndarray)
    assert bf_art.shape == (17,)
    assert bf_vein.shape == (17,)

    # Calculate expected blood flow
    xbf = bf_core + bf_muscle + bf_fat + bf_skin

    # Check that the elements in the array are correct
    assert np.isclose(bf_art[0], xbf[0])
    assert np.isclose(bf_vein[0], xbf[0])

    assert np.isclose(bf_art[1], xbf[1] + xbf[0])
    assert np.isclose(bf_vein[1], xbf[1] + xbf[0])

    assert np.isclose(bf_art[2], xbf[2])
    assert np.isclose(bf_vein[2], xbf[2])

    assert np.isclose(bf_art[3], xbf[3])
    assert np.isclose(bf_vein[3], xbf[3])


# test for thermoregulation.py
def test_conv_coef() -> None:
    """Test the conv_coef function for calculating convective coefficients based on posture."""
    # Test case 1: Default values
    hc_expected = np.asarray(
        [
            4.48,
            4.48,
            2.97,
            2.91,
            2.85,
            3.61,
            3.55,
            3.67,
            3.61,
            3.55,
            3.67,
            2.80,
            2.04,
            2.04,
            2.80,
            2.04,
            2.04,
        ],
    )
    assert np.array_equal(
        conv_coef(posture="standing", v=0.1, tdb=28.8, t_skin=34),
        hc_expected,
    )

    # Test case 2: Sitting posture
    hc_expected = np.asarray(
        [
            4.75,
            4.75,
            3.12,
            2.48,
            1.84,
            3.76,
            3.62,
            2.06,
            3.76,
            3.62,
            2.06,
            2.98,
            2.98,
            2.62,
            2.98,
            2.98,
            2.62,
        ],
    )
    assert np.array_equal(
        conv_coef(posture="sitting", tdb=28.8, v=0.1, t_skin=34.0),
        hc_expected,
    )

    # Test case 3: Invalid posture
    with pytest.raises(ValueError):
        conv_coef(posture="invalid", tdb=28.8, v=0.1, t_skin=34.0)

    # Test case 4: Lying posture with different tdb and t_skin values
    tdb = np.full(17, 28.8)
    t_skin = np.full(17, 34.0)
    hc_expected = np.asarray(
        [
            1.105,
            1.105,
            1.211,
            1.211,
            1.211,
            0.913,
            2.081,
            2.178,
            0.913,
            2.081,
            2.178,
            0.945,
            0.385,
            0.200,
            0.945,
            0.385,
            0.200,
        ],
    ) * (
        abs(tdb - t_skin)
        ** np.asarray(
            [
                0.345,
                0.345,
                0.046,
                0.046,
                0.046,
                0.373,
                0.850,
                0.297,
                0.373,
                0.850,
                0.297,
                0.447,
                0.580,
                0.966,
                0.447,
                0.580,
                0.966,
            ],
        )
    )
    assert np.allclose(
        conv_coef(posture="lying", tdb=tdb, t_skin=t_skin, v=0.1),
        hc_expected,
    )

    # Test case 5: Forced convection (v > 0.2)
    v = np.full(17, 0.3)
    hc_expected = np.asarray(
        [
            15.0,
            15.0,
            11.0,
            17.0,
            13.0,
            17.0,
            17.0,
            20.0,
            17.0,
            17.0,
            20.0,
            14.0,
            15.8,
            15.1,
            14.0,
            15.8,
            15.1,
        ],
    ) * (
        v
        ** np.asarray(
            [
                0.62,
                0.62,
                0.67,
                0.49,
                0.60,
                0.59,
                0.61,
                0.60,
                0.59,
                0.61,
                0.60,
                0.61,
                0.74,
                0.62,
                0.61,
                0.74,
                0.62,
            ],
        )
    )
    assert np.allclose(
        conv_coef(v=v, tdb=28.8, posture="standing", t_skin=34.0),
        hc_expected,
    )


def test_rad_coef() -> None:
    """Test the rad_coef function for calculating radiative coefficients based on posture."""
    # Test with valid postures
    valid_postures = {
        "standing": np.asarray(
            [
                4.89,
                4.89,
                4.32,
                4.09,
                4.32,
                4.55,
                4.43,
                4.21,
                4.55,
                4.43,
                4.21,
                4.77,
                5.34,
                6.14,
                4.77,
                5.34,
                6.14,
            ],
        ),
        "sitting": np.asarray(
            [
                4.96,
                4.96,
                3.99,
                4.64,
                4.21,
                4.96,
                4.21,
                4.74,
                4.96,
                4.21,
                4.74,
                4.10,
                4.74,
                6.36,
                4.10,
                4.74,
                6.36,
            ],
        ),
        "lying": np.asarray(
            [
                5.475,
                5.475,
                3.463,
                3.463,
                3.463,
                4.249,
                4.835,
                4.119,
                4.249,
                4.835,
                4.119,
                4.440,
                5.547,
                6.085,
                4.440,
                5.547,
                6.085,
            ],
        ),
        "sedentary": np.asarray(
            [
                4.96,
                4.96,
                3.99,
                4.64,
                4.21,
                4.96,
                4.21,
                4.74,
                4.96,
                4.21,
                4.74,
                4.10,
                4.74,
                6.36,
                4.10,
                4.74,
                6.36,
            ],
        ),
        "supine": np.asarray(
            [
                5.475,
                5.475,
                3.463,
                3.463,
                3.463,
                4.249,
                4.835,
                4.119,
                4.249,
                4.835,
                4.119,
                4.440,
                5.547,
                6.085,
                4.440,
                5.547,
                6.085,
            ],
        ),
    }
    for posture, expected in valid_postures.items():
        assert np.allclose(rad_coef(posture=posture), expected)

    # Test with invalid postures
    invalid_postures = ["invalid", "running", "jumping"]
    for posture in invalid_postures:
        with pytest.raises(ValueError):
            rad_coef(posture=posture)


def test_fixed_hc() -> None:
    """Test the fixed_hc function for calculating fixed convective heat transfer coefficient."""
    hc = np.ones(17) * 3
    v = np.ones(17) * 0.1

    # Call the fixed_hc function
    fixed_hc_values = fixed_hc(hc=hc, v=v)

    # Check if the function returns a numpy array
    assert isinstance(fixed_hc_values, np.ndarray)

    # Check if the function returns an array of the same shape as input
    assert fixed_hc_values.shape == hc.shape

    # Check if the function returns expected values (based on the formula)
    mean_hc = np.average(hc, weights=Default.local_bsa)
    mean_va = np.average(v, weights=Default.local_bsa)
    mean_hc_whole = max(3, 8.600001 * (mean_va**0.53))
    expected_fixed_hc = hc * mean_hc_whole / mean_hc
    assert np.allclose(fixed_hc_values, expected_fixed_hc)


def test_fixed_hr() -> None:
    """Test the fixed_hr function for calculating radiative coefficient."""
    hr = np.ones(17) * 5

    # Call the fixed_hr function
    fixed_hr_values = fixed_hr(hr)

    # Check if the function returns a numpy array
    assert isinstance(fixed_hr_values, np.ndarray)

    # Check if the function returns an array of the same shape as input
    assert fixed_hr_values.shape == hr.shape

    # Check if the function returns expected values (based on the formula)
    mean_hr = np.average(hr, weights=Default.local_bsa)
    expected_fixed_hr = hr * 4.7 / mean_hr
    assert np.allclose(fixed_hr_values, expected_fixed_hr)


def test_operative_temp() -> None:
    """Test the operative_temp function for calculating operative temperature."""
    # Test with scalar inputs
    tdb = 25.0
    tr = 25.0
    hc = 3
    hr = 5
    to = operative_temp(tdb, tr, hc, hr)
    expected_to = 25.0
    assert (
        pytest.approx(to) == expected_to
    )  # Since tdb and tr are the same, to should also be the same

    # Test with array inputs
    tdb = np.asarray([20.0, 22.0, 24.0])
    tr = np.asarray([25.0, 27.0, 29.0])
    hc = np.asarray([0.5, 0.6, 0.7])
    hr = np.asarray([0.5, 0.4, 0.3])
    to = operative_temp(tdb, tr, hc, hr)
    expected_to = (hc * tdb + hr * tr) / (hc + hr)
    assert np.allclose(to, expected_to)

    # Test with mixed scalar and array inputs
    tdb = 20.0
    tr = np.asarray([25.0, 27.0, 29.0])
    hc = 0.5
    hr = np.asarray([0.5, 0.4, 0.3])
    to = operative_temp(tdb, tr, hc, hr)
    expected_to = (hc * tdb + hr * tr) / (hc + hr)
    assert np.allclose(to, expected_to)


def test_clo_area_factor() -> None:
    """Test the clo_area_factor function for calculating clothing area factor."""
    # Test with single value less than 0.5
    clo = 0.4
    expected_result = 1.08
    assert clo_area_factor(clo) == pytest.approx(expected_result, rel=1e-3)

    # Test with single value less than 0.5
    clo = 0.6
    expected_result = 1.11
    assert clo_area_factor(clo) == pytest.approx(expected_result, rel=1e-3)

    # Test with zero value
    clo = 0
    expected_result = 1  # clothing area factor should be 1 when clo is 0
    assert clo_area_factor(clo) == pytest.approx(expected_result, rel=1e-3)

    # Test with array values
    clo = np.asarray([0.4, 0.6, 0.2])
    expected_result = np.asarray([1.08, 1.11, 1.04])
    np.testing.assert_allclose(clo_area_factor(clo), expected_result, rtol=1e-3)


def test_dry_r() -> None:
    """Test the dry_r function for calculating dry resistance."""
    # Test with single values
    hc, hr, clo = 3, 5, 0.5
    expected_result = 0.191
    assert dry_r(hc, hr, clo) == pytest.approx(expected_result, rel=1e-3)

    # Test with array values
    hc = np.asarray([3, 3])
    hr = np.asarray([5, 5])
    clo = np.asarray([0.5, 0.5])
    expected_result = np.asarray([0.191, 0.191])
    assert dry_r(hc, hr, clo) == pytest.approx(expected_result, rel=1e-3)

    # Test with zero values
    hc, hr, clo = 0, 0, 0
    with pytest.raises(ZeroDivisionError):
        dry_r(hc, hr, clo)

    # Test negative hc and hr values
    hc, hr, clo = -10.0, -5.0, 1.0
    with pytest.raises(ValueError):
        dry_r(hc, hr, clo)


def test_wet_r() -> None:
    """Test the wet_r function for calculating wet resistance."""
    # Test with single values
    hc = 10.0
    clo = 0.5
    i_clo = 0.45
    lewis_rate = 16.5
    expected_result = 0.01594
    assert wet_r(hc, clo, i_clo, lewis_rate) == pytest.approx(expected_result, rel=1e-3)

    # Test with array values
    hc = np.asarray([10.0, 10.0])
    clo = np.asarray([0.5, 0.5])
    i_clo = np.asarray([0.45, 0.45])
    lewis_rate = 16.5
    expected_result = np.asarray(
        [0.01594, 0.01594],
    )  # Replace with actual expected result based on your function's logic
    np.testing.assert_allclose(
        wet_r(hc, clo, i_clo, lewis_rate),
        expected_result,
        rtol=1e-3,
    )

    # Test with zero values
    hc, clo = 0, 0
    with pytest.raises(ZeroDivisionError):
        wet_r(hc, clo, i_clo, lewis_rate)

    # Test with zero value for clo
    hc, clo = -1, 0.5
    with pytest.raises(ValueError):
        wet_r(hc, clo, i_clo, lewis_rate)


def test_error_signals() -> None:
    """Test the error_signals function for calculating error signals."""
    # Test with default value:
    wrms, clds = error_signals()
    assert wrms == 0
    assert clds == 0

    # Test with positive value
    wrms, clds = error_signals(2)
    assert wrms > 0
    assert clds == 0

    # Test with negative value
    wrms, clds = error_signals(-2)
    assert wrms == 0
    assert clds > 0

    # Test with array
    err_sk = np.asarray(
        [-2, 2, -3, 3, -1, 1, 0, -0.5, 0.5, -2, 2, -1, 1, -0.5, 0.5, -1, 1],
    )
    wrms, clds = error_signals(err_sk)
    assert wrms > 0
    assert clds > 0

    # Test with wrong length array
    err_sk = np.asarray([-2, 2, -3])
    with pytest.raises(ValueError):
        error_signals(err_sk)


def test_evaporation() -> None:
    """Test the evaporation function for calculating evaporation rate."""
    # Test with basic parameters
    err_cr = np.asarray([0.5])
    err_sk = np.asarray(
        [
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            1.6,
            1.7,
            1.8,
        ],
    )
    t_skin = np.asarray([34.0] * 17)
    tdb = np.asarray([25.0] * 17)
    rh = np.asarray([50.0] * 17)
    ret = np.asarray([0.01] * 17)

    wet, e_sk, e_max, e_sweat = evaporation(
        err_cr,
        err_sk,
        t_skin,
        tdb,
        rh,
        ret,
        height=1.72,
        weight=74.43,
        bsa_equation="dubois",
        age=20,
    )

    assert isinstance(wet, np.ndarray)
    assert isinstance(e_sk, np.ndarray)
    assert isinstance(e_max, np.ndarray)
    assert isinstance(e_sweat, np.ndarray)
    assert np.all(wet >= 0)
    assert np.all(wet <= 1)

    # Test age effects
    err_cr = np.asarray([0.5])
    err_sk = np.asarray([0.2] * 17)
    t_skin = np.asarray([34.0] * 17)
    tdb = np.asarray([25.0] * 17)
    rh = np.asarray([50.0] * 17)
    ret = np.asarray([0.01] * 17)

    wet_young, e_sk_young, e_max_young, e_sweat_young = evaporation(
        err_cr,
        err_sk,
        t_skin,
        tdb,
        rh,
        ret,
        age=20,
        height=1.72,
        weight=74.43,
        bsa_equation="dubois",
    )
    wet_old, e_sk_old, e_max_old, e_sweat_old = evaporation(
        err_cr,
        err_sk,
        t_skin,
        tdb,
        rh,
        ret,
        age=65,
        height=1.72,
        weight=74.43,
        bsa_equation="dubois",
    )

    assert np.all(e_sweat_old <= e_sweat_young)

    # Test with valid BSA equations
    err_cr = np.asarray([0.5])
    err_sk = np.asarray([0.2] * 17)
    t_skin = np.asarray([34.0] * 17)
    tdb = np.asarray([25.0] * 17)
    rh = np.asarray([50.0] * 17)
    ret = np.asarray([0.01] * 17)

    valid_bsa_equations_list = ["dubois", "takahira", "fujimoto", "kurazumi"]
    for bsa_equation in valid_bsa_equations_list:
        wet, e_sk, e_max, e_sweat = evaporation(
            err_cr,
            err_sk,
            t_skin,
            tdb,
            rh,
            ret,
            bsa_equation=bsa_equation,
            height=1.72,
            weight=74.43,
            age=20,
        )

        assert isinstance(wet, np.ndarray)
        assert isinstance(e_sk, np.ndarray)
        assert isinstance(e_max, np.ndarray)
        assert isinstance(e_sweat, np.ndarray)

    # Test with invalid BSA equation
    invalid_bsa_equation = "sushi"
    with pytest.raises(ValueError):
        evaporation(
            err_cr,
            err_sk,
            t_skin,
            tdb,
            rh,
            ret,
            bsa_equation=invalid_bsa_equation,
            height=1.72,
            weight=74.43,
            age=20,
        )

    # Test to ensure no errors occur when e_max is zero
    t_skin = np.asarray([34.0] * 17)
    tdb = np.asarray([34.0] * 17)
    rh = np.asarray([100] * 17)

    # Call the evaporation function
    wet, e_sk, e_max, e_sweat = evaporation(
        err_cr,
        err_sk,
        t_skin,
        tdb,
        rh,
        ret,
        height=1.72,
        weight=74.43,
        bsa_equation="dubois",
        age=20,
    )

    expected_e_max = 0.001
    expected_wet = 1
    # Check that all elements in e_max have been replaced with 0.001
    assert np.all(
        e_max == expected_e_max,
    )  # Verify that e_max has been replaced by 0.001
    assert np.all(
        wet == pytest.approx(expected_wet, rel=1e-3),
    )  # Verify that wet is nealy 1


def test_skin_blood_flow() -> None:
    """Test the skin_blood_flow function for calculating blood flow in the skin."""
    # Test with basic values
    err_cr = np.asarray([0.5])
    err_sk = np.asarray(
        [
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            1.6,
            1.7,
            1.8,
        ],
    )

    bf_skin = skin_blood_flow(
        err_cr,
        err_sk,
        height=1.72,
        weight=74.43,
        bsa_equation="dubois",
        age=20,
        ci=2.59,
    )

    assert isinstance(bf_skin, np.ndarray)
    assert np.all(bf_skin >= 0)

    # Test age effect
    err_cr = np.asarray([0.5])
    err_sk = np.asarray([0.2] * 17)

    bf_skin_young = skin_blood_flow(
        err_cr,
        err_sk,
        age=20,
        height=1.72,
        weight=74.43,
        bsa_equation="dubois",
        ci=2.59,
    )
    bf_skin_old = skin_blood_flow(
        err_cr,
        err_sk,
        age=65,
        height=1.72,
        weight=74.43,
        bsa_equation="dubois",
        ci=2.59,
    )

    assert np.all(bf_skin_old <= bf_skin_young)

    # Test valid BSA equation
    valid_bsa_equations_list = ["dubois", "takahira", "fujimoto", "kurazumi"]
    for bsa_equation in valid_bsa_equations_list:
        err_cr = np.asarray([0.5])
        err_sk = np.asarray([0.2] * 17)

        bf_skin: np.ndarray = skin_blood_flow(
            err_cr,
            err_sk,
            bsa_equation=bsa_equation,
            height=1.72,
            weight=74.43,
            age=20,
            ci=2.59,
        )

        assert isinstance(bf_skin, np.ndarray)
        assert np.all(bf_skin >= 0)

    # Test with invalid BSA equation
    invalid_bsa_equation = "sushi"
    with pytest.raises(ValueError):
        skin_blood_flow(
            err_cr,
            err_sk,
            bsa_equation=invalid_bsa_equation,
            height=1.72,
            weight=74.43,
            age=20,
            ci=2.59,
        )


def test_ava_blood_flow() -> None:
    """Test the ava_blood_flow function for calculating blood flow in arteriovenous anastomoses (AVA)."""
    # Test with basic parameters
    err_cr = np.asarray([0.5, 0.6, 0.7, 0.8, 0.9])
    err_sk = np.asarray(
        [
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            1.6,
            1.7,
            1.8,
        ],
    )

    bf_ava_hand, bf_ava_foot = ava_blood_flow(
        err_cr,
        err_sk,
        height=1.72,
        weight=74.43,
        bsa_equation="dubois",
        age=20,
        ci=2.59,
    )

    assert isinstance(bf_ava_hand, float)
    assert isinstance(bf_ava_foot, float)
    assert bf_ava_hand >= 0
    assert bf_ava_foot >= 0

    # Test age effect
    err_cr = np.asarray([0.5, 0.6, 0.7, 0.8, 0.9])
    err_sk = np.asarray([0.2] * 17)

    bf_ava_hand_young, bf_ava_foot_young = ava_blood_flow(
        err_cr,
        err_sk,
        age=20,
        height=1.72,
        weight=74.43,
        bsa_equation="dubois",
        ci=2.59,
    )
    bf_ava_hand_old, bf_ava_foot_old = ava_blood_flow(
        err_cr,
        err_sk,
        age=65,
        height=1.72,
        weight=74.43,
        bsa_equation="dubois",
        ci=2.59,
    )

    assert bf_ava_hand_old <= bf_ava_hand_young
    assert bf_ava_foot_old <= bf_ava_foot_young

    # Test with input length mismatch
    err_cr = np.asarray([0.5, 0.6, 0.7])
    err_sk = np.asarray([0.2, 0.3])
    with pytest.raises(TypeError):
        ava_blood_flow(
            err_cr,
            err_sk,
            height=1.72,
            weight=74.43,
            bsa_equation="dubois",
            age=20,
            ci=2.59,
        )

    # Test with zero signal
    err_cr = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0])
    err_sk = np.asarray([0.0] * 17)
    bf_ava_hand, bf_ava_foot = ava_blood_flow(
        err_cr,
        err_sk,
        height=1.72,
        weight=74.43,
        bsa_equation="dubois",
        age=20,
        ci=2.59,
    )

    # (Need to be investigated)
    # Expected results are 1.71258 for hands and 1.42223 for feet when focusing on the
    # equation (Takemori, 1995.), but they might be 0 considering
    # characteristics of ava blood flow.
    expected_result_hand = 1.71258
    expected_result_foot = 1.42223

    assert bf_ava_hand == pytest.approx(expected_result_hand, rel=1e-3)
    assert bf_ava_foot == pytest.approx(expected_result_foot, rel=1e-3)


def test_basal_met() -> None:
    """Test the basal_met function for calculating basal metabolic rate (BMR)."""
    # Test with default values
    bmr = basal_met(
        height=1.72,
        weight=74.43,
        age=20,
        sex="male",
        bmr_equation="harris-benedict",
    )
    expected_result = 87.95
    assert bmr == pytest.approx(expected_result, rel=1e-3)

    # Test with custom values
    bmr = basal_met(
        height=1.80,
        weight=70,
        age=25,
        sex="male",
        bmr_equation="harris-benedict",
    )
    expected_result = 85.66
    assert bmr == pytest.approx(expected_result, rel=1e-3)

    # Test with valid BMR equations
    valid_equations_list = [
        "harris-benedict",
        "harris-benedict_origin",
        "japanese",
        "ganpule",
    ]
    for valid_equation in valid_equations_list:
        bmr = basal_met(
            bmr_equation=valid_equation,
            height=1.72,
            weight=74.4,
            age=20,
            sex="male",
        )
        assert isinstance(bmr, float)

    # Test with invalid BMR equation
    invalid_bsa_equation = "sushi"
    with pytest.raises(ValueError):
        basal_met(
            bmr_equation=invalid_bsa_equation,
            height=1.72,
            weight=74.43,
            age=20,
            sex="male",
        )


def test_local_mbase() -> None:
    """Test the local_mbase function for calculating local metabolic rates."""
    # Test with default values
    mbase_cr, mbase_ms, mbase_fat, mbase_sk = local_mbase()
    assert isinstance(mbase_cr, np.ndarray)
    assert isinstance(mbase_ms, np.ndarray)
    assert isinstance(mbase_fat, np.ndarray)
    assert isinstance(mbase_sk, np.ndarray)

    # Test with custom values
    mbase_cr, mbase_ms, mbase_fat, mbase_sk = local_mbase(
        height=1.80,
        weight=70,
        age=25,
        sex="male",
        bmr_equation="harris-benedict",
    )

    # Check that each element of mbase_cr is greater than the corresponding elements
    # of mbase_ms, mbase_fat, and mbase_sk
    assert all(mbase_cr > mbase_ms)
    assert all(mbase_cr > mbase_fat)
    assert all(mbase_cr > mbase_sk)


def test_local_q_work() -> None:
    """Test the local_q_work function for calculating local work rate."""
    # Test with par = 1.5
    q_work = local_q_work(bmr=100, par=1.5)
    assert all(q_work >= 0)
    # Test with par = 1.0
    q_work = local_q_work(bmr=100, par=1.0)
    assert all(q_work == 0)
    # Test with par < 1.0
    with pytest.raises(ValueError):
        local_q_work(bmr=100, par=0.8)


def test_shivering() -> None:
    """Test the shivering function for calculating shivering thermogenesis."""
    # Test with zero error signals
    err_cr = np.zeros(17)
    err_sk = np.zeros(17)
    t_core = np.ones(17) * 36
    t_skin = np.ones(17) * 32
    q_shiv = shivering(
        err_cr,
        err_sk,
        t_core,
        t_skin,
        height=1.72,
        weight=74.43,
        age=20,
        sex="male",
        dtime=1,
        options=None,
        bsa_equation="dubois",
    )
    assert all(q_shiv == 0)

    # Test with positive error signals
    err_cr = np.ones(17) * 1
    err_sk = np.ones(17) * 1
    t_core = np.ones(17) * 36
    t_skin = np.ones(17) * 32
    q_shiv = shivering(
        err_cr,
        err_sk,
        t_core,
        t_skin,
        height=1.72,
        weight=74.43,
        age=20,
        sex="male",
        dtime=1,
        options=None,
        bsa_equation="dubois",
    )
    assert all(q_shiv == 0)

    # Test positive shivering thermogenesis with negative error signals
    err_cr = np.ones(17) * -1
    err_sk = np.ones(17) * -2
    t_core = np.ones(17) * 36
    t_skin = np.ones(17) * 32
    q_shiv = shivering(
        err_cr,
        err_sk,
        t_core,
        t_skin,
        height=1.72,
        weight=74.43,
        age=20,
        sex="male",
        dtime=1,
        options=None,
        bsa_equation="dubois",
    )
    assert all(q_shiv > 0)

    # Test with different age group
    err_cr = np.ones(17) * -1
    err_sk = np.ones(17) * -2
    t_core = np.ones(17) * 36
    t_skin = np.ones(17) * 32

    age_list = [25, 35, 45, 55, 65, 75, 85]
    q_shiv_by_age = {}

    for age in age_list:
        q_shiv = shivering(
            err_cr,
            err_sk,
            t_core,
            t_skin,
            age=age,
            height=1.72,
            weight=74.43,
            sex="male",
            dtime=1,
            options=None,
            bsa_equation="dubois",
        )
        q_shiv_by_age[age] = q_shiv

    # Compare the results for different ages
    for i in range(len(age_list) - 1):
        age_younger = age_list[i]
        age_older = age_list[i + 1]
        assert all(q_shiv_by_age[age_younger] > q_shiv_by_age[age_older])


def test_nonshivering() -> None:
    """Test the nonshivering function for calculating non-shivering thermogenesis."""
    # Test with zero error signals
    err_sk = np.zeros(17)
    q_nst = nonshivering(
        err_sk,
        height=1.72,
        weight=74.43,
        age=20,
        bsa_equation="dubois",
        cold_acclimation=False,
        batpositive=True,
    )
    assert all(q_nst == 0)

    # Test with zero error signals
    err_sk = np.ones(17) * 1
    q_nst = nonshivering(
        err_sk,
        height=1.72,
        weight=74.43,
        age=20,
        bsa_equation="dubois",
        cold_acclimation=False,
        batpositive=True,
    )
    assert all(q_nst == 0)

    # Test with negative error signals
    err_sk = np.ones(17) * -1
    q_nst = nonshivering(
        err_sk,
        height=1.72,
        weight=74.43,
        age=20,
        bsa_equation="dubois",
        cold_acclimation=False,
        batpositive=True,
    )
    assert all(q_nst >= 0)

    # Test age effect on BAT (brown adipose tissue) that affects NST limit
    err_sk = np.ones(17) * -10  # Set -10 to check the NST limit is working
    age_list = [25, 35, 45]
    q_nst_by_age = {}
    sum_q_nst_by_age = {}

    for age in age_list:
        q_nst = nonshivering(
            err_sk,
            age=age,
            height=1.72,
            weight=74.43,
            bsa_equation="dubois",
        )
        q_nst_by_age[age] = q_nst
        sum_q_nst_by_age[age] = np.sum(q_nst_by_age[age])

    # Compare the results for different ages
    for i in range(len(age_list) - 1):
        age_younger = age_list[i]
        age_older = age_list[i + 1]
        assert sum_q_nst_by_age[age_younger] > sum_q_nst_by_age[age_older]


def test_cold_acclimation_non_shivering() -> None:
    """Test the cold acclimation effect on non-shivering thermogenesis."""
    # Test cold acclimation that affects NST limit
    err_sk = np.ones(17) * -10  # Set -10 to check the NST limit is working
    q_nst_no_acclimation = nonshivering(
        err_sk,
        cold_acclimation=False,
        batpositive=True,
        height=1.72,
        weight=74.43,
        age=20,
        bsa_equation="dubois",
    )
    q_nst_with_acclimation = nonshivering(
        err_sk,
        cold_acclimation=True,
        batpositive=True,
        height=1.72,
        weight=74.43,
        age=20,
        bsa_equation="dubois",
    )
    assert not np.array_equal(q_nst_no_acclimation, q_nst_with_acclimation)


def test_sum_bf() -> None:
    """Test the sum_bf function for calculating body fat."""
    # Test to check output type
    bf_core = np.asarray([1, 2, 3, 4])
    bf_muscle = np.asarray([1, 2, 3, 4])
    bf_fat = np.asarray([1, 2, 3, 4])
    bf_skin = np.asarray([1, 2, 3, 4])
    bf_ava_hand = 1.0
    bf_ava_foot = 1.0

    co = sum_bf(bf_core, bf_muscle, bf_fat, bf_skin, bf_ava_hand, bf_ava_foot)
    assert isinstance(co, float)

    # Test with custom values 1
    bf_core = np.asarray([1, 2, 3, 4])
    bf_muscle = np.asarray([1, 2, 3, 4])
    bf_fat = np.asarray([1, 2, 3, 4])
    bf_skin = np.asarray([1, 2, 3, 4])
    bf_ava_hand = 1.0
    bf_ava_foot = 1.0

    co = sum_bf(bf_core, bf_muscle, bf_fat, bf_skin, bf_ava_hand, bf_ava_foot)
    expected_co = (
        bf_core.sum()
        + bf_muscle.sum()
        + bf_fat.sum()
        + bf_skin.sum()
        + 2 * bf_ava_hand
        + 2 * bf_ava_foot
    )
    assert co == expected_co

    # Test with custom values 2
    bf_core = np.asarray([1, 2, 3, 4])
    bf_muscle = np.asarray([1, 2, 3, 4])
    bf_fat = np.asarray([1, 2, 3, 4])
    bf_skin = np.asarray([1, 2, 3, 4])
    bf_ava_hand = 1.0
    bf_ava_foot = 1.0

    bf_core_copy = bf_core.copy()
    bf_muscle_copy = bf_muscle.copy()
    bf_fat_copy = bf_fat.copy()
    bf_skin_copy = bf_skin.copy()

    sum_bf(bf_core, bf_muscle, bf_fat, bf_skin, bf_ava_hand, bf_ava_foot)

    assert np.array_equal(bf_core, bf_core_copy)
    assert np.array_equal(bf_muscle, bf_muscle_copy)
    assert np.array_equal(bf_fat, bf_fat_copy)
    assert np.array_equal(bf_skin, bf_skin_copy)


def test_resp_heat_loss() -> None:
    """Test the resp_heat_loss function for calculating respiratory heat loss."""
    # Test to check output type
    tdb = 25.0
    p_a = 1.0
    q_thermogenesis_total = 100.0

    res_sh, res_lh = resp_heat_loss(tdb, p_a, q_thermogenesis_total)
    assert isinstance(res_sh, float)
    assert isinstance(res_lh, float)

    # Test to check correct heat loss
    tdb = 25.0
    p_a = 1.0
    q_thermogenesis_total = 100.0

    res_sh, res_lh = resp_heat_loss(tdb, p_a, q_thermogenesis_total)
    expected_res_sh = 0.0014 * q_thermogenesis_total * (34 - tdb)
    expected_res_lh = 0.0173 * q_thermogenesis_total * (5.87 - p_a)

    assert res_sh == expected_res_sh
    assert res_lh == expected_res_lh

    # Test sensible heat loss at 34oC of air temp
    tdb = 34.0
    p_a = 1.0
    q_thermogenesis_total = 100.0

    res_sh, _ = resp_heat_loss(tdb, p_a, q_thermogenesis_total)
    assert res_sh == 0.0


def test_dict_results_returns_values_not_body_part_names() -> None:
    """Per-segment columns must hold simulated values, not the segment name.

    Regression test for issue #264: dict_results() zipped its column keys
    against a JOS3BodyParts' __dict__, and iterating a dict yields its keys, so
    every per-segment column held its own body part name as a string.
    """
    model = JOS3(height=1.7, weight=60, age=30)
    model.to = 28
    model.rh = 40
    model.v = 0.1
    model.par = 1.0
    model.simulate(3)

    results = model.dict_results()

    # The reported bug: the column held the string "head".
    assert not isinstance(results["t_skin_head"][-1], str)
    assert isinstance(results["t_skin_head"][-1], (int, float, np.floating))

    # "sex" is the only column that is legitimately a string.
    string_columns = [
        key
        for key, series in results.items()
        if any(isinstance(entry, str) for entry in series)
    ]
    assert string_columns == ["sex"], (
        f"unexpected string-valued columns: {string_columns}"
    )


def test_dict_results_subset_variables_align_with_results() -> None:
    """Variables defined on a subset of segments must pick the right segments.

    t_muscle and t_fat exist only for head and pelvis, but are carried in a
    full 17-segment container. Selecting the first len(keys) values rather than
    the VINDEX-indicated ones paired t_muscle_pelvis with the neck's value.
    """
    model = JOS3(height=1.7, weight=60, age=30)
    model.to = 28
    model.rh = 40
    model.v = 0.1
    model.par = 1.0
    model.simulate(3)

    flat = model.dict_results()
    structured = model.results()

    assert flat["t_muscle_head"][-1] == structured.t_muscle.head[-1]
    assert flat["t_muscle_pelvis"][-1] == structured.t_muscle.pelvis[-1]
    assert flat["t_fat_head"][-1] == structured.t_fat.head[-1]
    assert flat["t_fat_pelvis"][-1] == structured.t_fat.pelvis[-1]
