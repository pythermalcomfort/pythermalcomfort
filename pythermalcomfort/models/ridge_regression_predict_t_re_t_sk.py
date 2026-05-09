"""Ridge regression model for predicting rectal and skin temperature.

Based on fold 1 from Forbes et al. (2025), doi:10.1016/j.jtherbio.2025.104078. Trained
on adults aged 60-100 in minimal clothing under stationary conditions. Uses Min-Max
scaling on features and outputs.
"""

from __future__ import annotations

import numpy as np

from pythermalcomfort.classes_input import RidgeRegressionInputs
from pythermalcomfort.classes_return import PredictedBodyTemperatures
from pythermalcomfort.shared_functions import valid_range
from pythermalcomfort.utilities import Sex

# --- Model Constants ---

# Minimum values for each of the 8 input features, used for Min-Max scaling.
_FEATURES_SCALER_OFFSET = np.array(
    [
        0.0,  # sex
        -0.3114754098360656,  # age
        -3.020833333333333,  # height
        -0.6183587248751761,  # weight
        -1.222222222222222,  # tdb
        -0.21951219512195122,  # rh
        -11.197107405358395,  # t_re
        -2.105553500973682,  # t_sk
    ]
)

# Scaling factors (1 / (max - min)) for each of the 8 input features, used for Min-Max scaling.
_FEATURES_SCALER_SCALE = np.array(
    [
        1.0,  # sex
        0.01639344262295082,  # age
        0.020833333333333332,  # height
        0.012802458071949815,  # weight
        0.05555555555555555,  # tdb
        0.024390243902439025,  # rh
        0.31613056192207334,  # t_re
        0.08119094120192633,  # t_sk
    ]
)

# Minimum values for the 2 output variables (Rectal Temp, Mean Skin Temp), used for inverse scaling.
_OUTPUT_SCALER_OFFSET = np.array(
    [
        -11.197107405358395,  # t_re
        -2.0197777680408033,  # t_sk
    ]
)

# Scaling factors (1 / (max - min)) for the 2 output variables, used for inverse scaling.
_OUTPUT_SCALER_SCALE = np.array(
    [
        0.31613056192207334,  # t_re
        0.07894843838015173,  # t_sk
    ]
)

# Coefficients for the Rectal Temperature (t_re) ridge regression model.
# Corresponds to the 8 input features in order.
_T_RE_COEFFS = np.array(
    [
        0.00016261586852849347,  # sex
        0.0007368142143779594,  # age
        -0.00043916987857211637,  # height
        0.00046532701146677997,  # weight
        0.0008443934806620367,  # tdb
        0.0006663379066237714,  # rh
        0.9932810428489056,  # t_re (previous)
        0.006016233208250791,  # t_sk (previous)
    ]
)
# Intercept term for the Rectal Temperature (t_re) ridge regression model.
_T_RE_INTERCEPT = -0.0013528489525256315

# Coefficients for the Mean Skin Temperature (t_sk) ridge regression model.
# Corresponds to the 8 input features in order.
_T_SK_COEFFS = np.array(
    [
        0.0006157845452869151,  # sex
        0.00014854705372386215,  # age
        -0.0004329826169348138,  # height
        -0.0011471088118388912,  # weight
        0.018904677058503336,  # tdb
        0.003188995712763656,  # rh
        -0.0010477636196332153,  # t_re (previous)
        0.933918210580563,  # t_sk (previous)
    ]
)
# Intercept term for the Mean Skin Temperature (t_sk) ridge regression model.
_T_SK_INTERCEPT = 0.04356328728329839

# Thermoneutral baseline configuration
_BASELINE_DURATION = 120  # minutes
_THERMONEUTRAL_TDB = 23.0  # °C
_THERMONEUTRAL_RH = 50.0  # %
_BASELINE_T_RE_INITIAL = 37.0  # °C, initial rectal temperature for baseline
_BASELINE_T_SK_INITIAL = 32.0  # °C, initial skin temperature for baseline


def _scale_features(features: np.ndarray) -> np.ndarray:
    """Scales input features using predefined scaling constants."""
    return features * _FEATURES_SCALER_SCALE + _FEATURES_SCALER_OFFSET


def _inverse_scale_output(scaled_output: np.ndarray) -> np.ndarray:
    """Apply inverse scaling to the model output."""
    return (scaled_output - _OUTPUT_SCALER_OFFSET) / _OUTPUT_SCALER_SCALE


def _predict_temperature_simulation(
    features: np.ndarray, duration: int
) -> tuple[np.ndarray, np.ndarray]:
    """Core simulation loop for temperature prediction."""
    # Scale input features
    scaled_features = _scale_features(features)

    # Precompute static components of the regression
    static_t_re = np.dot(scaled_features[:, :6], _T_RE_COEFFS[:6]) + _T_RE_INTERCEPT
    static_t_sk = np.dot(scaled_features[:, :6], _T_SK_COEFFS[:6]) + _T_SK_INTERCEPT

    # Initialize temperatures from scaled features
    prev_t_re = scaled_features[:, 6]
    prev_t_sk = scaled_features[:, 7]

    # Store history of temperatures
    n_scenarios = features.shape[0]
    t_re_history_scaled = np.empty((n_scenarios, duration), dtype=float)
    t_sk_history_scaled = np.empty((n_scenarios, duration), dtype=float)

    # Run simulation
    for i in range(duration):
        new_t_re = (
            static_t_re + _T_RE_COEFFS[6] * prev_t_re + _T_RE_COEFFS[7] * prev_t_sk
        )
        new_t_sk = (
            static_t_sk + _T_SK_COEFFS[6] * prev_t_re + _T_SK_COEFFS[7] * prev_t_sk
        )
        prev_t_re, prev_t_sk = new_t_re, new_t_sk

        t_re_history_scaled[:, i] = new_t_re
        t_sk_history_scaled[:, i] = new_t_sk

    # Stack as (n_scenarios, duration, 2) and inverse-scale via broadcasting
    stacked = np.stack([t_re_history_scaled, t_sk_history_scaled], axis=-1)
    inv = _inverse_scale_output(stacked)
    t_re_history = inv[..., 0]
    t_sk_history = inv[..., 1]

    return t_re_history, t_sk_history


def _check_ridge_regression_compliance(
    age: np.ndarray,
    height: np.ndarray,
    weight: np.ndarray,
    tdb: np.ndarray,
    rh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Check if the inputs are within the model's applicability limits.

    Parameters
    ----------
    age : np.ndarray
        Age in years.
    height : np.ndarray
        Body height in cm.
    weight : np.ndarray
        Body weight in kg.
    tdb : np.ndarray
        Ambient air temperature in °C.
    rh : np.ndarray
        Relative humidity in %.

    Returns
    -------
    tuple of np.ndarray
        Five arrays with valid values preserved and out-of-range values set to NaN:
        (age_valid, height_valid, weight_valid, temp_valid, rh_valid).
    """
    age_valid = valid_range(age, (60, 100), "age")
    height_valid = valid_range(height, (130, 230), "height")
    weight_valid = valid_range(weight, (40, 140), "weight")
    temp_valid = valid_range(tdb, (0, 60), "tdb")
    rh_valid = valid_range(rh, (0, 100), "rh")
    return age_valid, height_valid, weight_valid, temp_valid, rh_valid


def ridge_regression_predict_t_re_t_sk(
    sex: Sex | str | list[Sex | str],
    age: float | list[float],
    height: float | list[float],
    weight: float | list[float],
    tdb: float | list[float],
    rh: float | list[float],
    duration: int,
    t_re: float | list[float] | None = None,
    t_sk: float | list[float] | None = None,
    limit_inputs: bool = True,
    round_output: bool = True,
) -> PredictedBodyTemperatures:
    """Predicts the full history of core and skin temperature changes based on a ridge
    regression model.

    This model simulates the rectal (t_re) and mean skin (t_sk)
    temperatures by first establishing a baseline in a thermoneutral environment
    (23°C, 50% RH for 120 mins) and then simulating exposure to the specified
    environmental conditions for `duration`. If a `t_re` and
    `t_sk` are provided, then the initial 120 min simulation is skipped.
    Coefficients are taken from fold one from the study [Forbes2025]_
    (https://doi.org/10.1016/j.jtherbio.2025.104078). See notes documentation
    for limitations with this model.

    Parameters
    ----------
    sex : Sex or str or list of (Sex or str)
        Biological sex. Pass the enum instance (e.g., `Sex.male`),
        its string value (e.g., `Sex.male.value` or `"male"`),
        or a list of either for vectorized inputs.
    age : float or list
        Age, in years.
    height : float or list
        Body height in meters [m].
    weight : float or list
        Body weight in kilograms [kg].
    tdb : float or list
        Ambient (dry bulb) air temperature, in °C.
    rh : float or list
        Relative humidity, in %.
    duration : int
        Duration of the simulation in the specified environment, in minutes.
    t_re : float or list, optional
        Initial rectal temperature (°C). If provided, the baseline simulation
        at 23°C, 50% RH for 120 minutes is skipped, and this value is used
        as the starting rectal temperature for the main simulation.
    t_sk : float or list, optional
        Initial mean skin temperature (°C). If provided, the baseline simulation
        at 23°C, 50% RH for 120 minutes is skipped, and this value is used
        as the starting mean skin temperature for the main simulation.
    limit_inputs : bool, optional
        If True, limits the inputs to the standard applicability limits. Defaults to True.
    round_output : bool, optional
        If True, rounds outputs to 2 decimal places; otherwise leaves full precision.
        Defaults to True.

    Returns
    -------
    PredictedBodyTemperatures
        A dataclass containing the predicted rectal (`t_re`) and skin (`t_sk`)
        temperature history in °C. See
        :py:class:`~pythermalcomfort.classes_return.PredictedBodyTemperatures`
        for more details. The outputs are numpy arrays. For scalar inputs, the shape
        is (`duration`,). For vector inputs, the shape is (`n_inputs`, `duration`).

    Raises
    ------
    ValueError
        If input arrays have inconsistent shapes after broadcasting.
        If sex contains invalid values (must be "male"/"female" or Sex enum members).
        If only one of t_re or t_sk is provided.
        If t_re and t_sk are not broadcastable to the input shape.

    Notes
    -----
    The model is applicable for adults aged between 60 and 100 years. The ranges for
    the input parameters are:

    - **Age**: 60 to 100 years
    - **Height**: 1.30 to 2.30 m
    - **Weight**: 40 to 140 kg
    - **Ambient Temperature**: 0 to 60 °C
    - **Relative Humidity**: 0 to 100 %

    If inputs fall outside applicability limits and `limit_inputs=True`, the
    corresponding output time series are filled with NaN for all time steps.

    The model does not have inputs such as: air velocity, radiative heat transfer,
    clothing level and an individual's activity level due to there being little
    variation in the dataset.

    All individuals in the dataset were in minimal clothing while stationary in a
    heat chamber, therefore stationary individuals in minimal clothing are best
    represented by this model.

    The function supports vectorized inputs for all parameters, allowing for
    batch predictions.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.utilities import Sex
        from pythermalcomfort.models.ridge_regression_predict_t_re_t_sk import (
            ridge_regression_predict_t_re_t_sk,
        )

        # Scalar example for a single person
        results = ridge_regression_predict_t_re_t_sk(
            sex=Sex.male.value,
            age=60,
            height=1.8,
            weight=75,
            tdb=35,
            rh=60,
            duration=540,
        )

        print(f"Final Rectal temp: {results.t_re[-1]:.2f}°C")
        print(f"Final Skin temp: {results.t_sk[-1]:.2f}°C")

        # Vectorized example for multiple scenarios
        results_vec = ridge_regression_predict_t_re_t_sk(
            sex=[Sex.male.value, Sex.female.value],
            age=[60, 65],
            height=[1.8, 1.65],
            weight=[75, 60],
            tdb=[35, 40],
            rh=[60, 50],
            duration=540,
        )

        print(f"Final rectal temps: {results_vec.t_re[:, -1]}")
        print(f"Rectal temp history shape: {results_vec.t_re.shape}")
    """
    # Validate inputs
    RidgeRegressionInputs(
        sex=sex,
        age=age,
        height=height,
        weight=weight,
        tdb=tdb,
        rh=rh,
        duration=duration,
        t_re=t_re,
        t_sk=t_sk,
        limit_inputs=limit_inputs,
        round_output=round_output,
    )

    # Convert height from m to cm, handling both scalar and list-like inputs
    height_cm = np.asarray(height) * 100

    # Convert sex (enum or string) to string values and then 0/1 encoding
    sex_arr = np.atleast_1d(sex)
    sex_str = np.array(
        [(s.value if isinstance(s, Sex) else str(s)).strip().lower() for s in sex_arr]
    )
    valid_sex_values = np.array([Sex.male.value, Sex.female.value])
    if not np.all(np.isin(sex_str, valid_sex_values)):
        err_msg = f"Invalid input for sex. Must be one of {valid_sex_values.tolist()}"
        raise ValueError(err_msg)
    # 1 for female, 0 for male
    sex_value_arr = (sex_str == Sex.female.value).astype(int)
    # If the original input was a scalar, convert the array back to a scalar
    # to preserve the correct output shape.
    supported_sex_values = (list, tuple, np.ndarray)
    if not isinstance(sex, supported_sex_values):
        sex_value = sex_value_arr.item()
    else:
        sex_value = sex_value_arr

    # Convert inputs to numpy arrays for vectorization
    inputs = np.broadcast_arrays(sex_value, age, height_cm, weight, tdb, rh)
    original_shape = inputs[0].shape
    flat_inputs = [np.ravel(i) for i in inputs]

    if limit_inputs:
        (
            age_valid,
            height_valid,
            weight_valid,
            temp_valid,
            rh_valid,
        ) = _check_ridge_regression_compliance(
            flat_inputs[1],
            flat_inputs[2],
            flat_inputs[3],
            flat_inputs[4],
            flat_inputs[5],
        )

    if t_re is not None and t_sk is not None:
        try:
            # Use provided baseline temperatures
            current_t_re = np.broadcast_to(np.asarray(t_re), original_shape).ravel()
            current_t_sk = np.broadcast_to(np.asarray(t_sk), original_shape).ravel()
        except ValueError as err:
            message = (
                "t_re and t_sk must be broadcastable to the input shape "
                f"{original_shape}."
            )
            raise ValueError(message) from err
    else:
        # Baseline simulation (120 mins in a neutral environment)
        t_re = np.full_like(flat_inputs[0], _BASELINE_T_RE_INITIAL, dtype=float)
        t_sk = np.full_like(flat_inputs[0], _BASELINE_T_SK_INITIAL, dtype=float)

        baseline_features = np.stack(
            [
                flat_inputs[0],  # sex
                flat_inputs[1],  # age
                flat_inputs[2],  # height
                flat_inputs[3],  # weight
                np.full_like(flat_inputs[0], _THERMONEUTRAL_TDB),  # tdb
                np.full_like(flat_inputs[0], _THERMONEUTRAL_RH),  # relative humidity
                t_re,
                t_sk,
            ],
            axis=1,
        )

        baseline_t_re_hist, baseline_t_sk_hist = _predict_temperature_simulation(
            baseline_features, duration=_BASELINE_DURATION
        )
        current_t_re = baseline_t_re_hist[:, -1]
        current_t_sk = baseline_t_sk_hist[:, -1]

    # Final simulation in specified environment
    final_features = np.stack(
        [
            flat_inputs[0],  # sex
            flat_inputs[1],  # age
            flat_inputs[2],  # height
            flat_inputs[3],  # weight
            flat_inputs[4],  # tdb
            flat_inputs[5],  # humidity
            current_t_re,
            current_t_sk,
        ],
        axis=1,
    )

    final_t_re_hist, final_t_sk_hist = _predict_temperature_simulation(
        final_features,
        duration=duration,
    )

    if limit_inputs:
        all_valid = ~(
            np.isnan(age_valid)
            | np.isnan(height_valid)
            | np.isnan(weight_valid)
            | np.isnan(temp_valid)
            | np.isnan(rh_valid)
        )
        # Apply validity mask across the time dimension
        validity_mask = all_valid[:, np.newaxis]
        final_t_re_hist = np.where(validity_mask, final_t_re_hist, np.nan)
        final_t_sk_hist = np.where(validity_mask, final_t_sk_hist, np.nan)

    if round_output:
        final_t_re_hist = np.round(final_t_re_hist, 2)
        final_t_sk_hist = np.round(final_t_sk_hist, 2)

    # If original input was scalar, return 1D array instead of 2D
    if not original_shape:
        final_t_re = final_t_re_hist.squeeze()
        final_t_sk = final_t_sk_hist.squeeze()
    else:
        final_t_re = final_t_re_hist
        final_t_sk = final_t_sk_hist

    return PredictedBodyTemperatures(t_re=final_t_re, t_sk=final_t_sk)
