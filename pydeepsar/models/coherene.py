"""
Module for computing coherence measures using TensorFlow.

This module provides functions for computing coherence measures
using TensorFlow, a popular deep learning framework.


"""

from typing import Optional

import numpy as np
import pandas as pd


def create_model_input_output(
    dataframe: pd.DataFrame, output: Optional[dict[str, str]] = None
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Create input X with z_repeated and z0_tensor.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame containing the input data.
    output : dict, optional
        Dictionary containing the output column names as keys.

    Returns
    -------
    tuple
        Tuple containing the inputs dictionary and optional output dictionary.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create a sample DataFrame
    >>> df = pd.DataFrame({
    ...     'geo_kz_ml': [0.1, 0.2, 0.3],
    ...     'geo_thetainc_ml': [0.2, 0.3, 0.4],
    ...     'geo_amp': [0.5, 0.6, 0.7],
    ...     'geo_coh': [0.8, 0.9, 1.0],
    ...     'geo_pha': [1.1, 1.2, 1.3],
    ... })
    >>> # Create output dictionary
    >>> output_dict = {'output1': 'humidity', 'output2': 'wind_speed'}
    >>> X, y = create_model_input_output(df, output_dict)
    """
    # Copy the DataFrame to data
    data = dataframe.copy()

    # Define z values
    a_input = -500.0
    b_input = 0.0
    num_intervals_input = 1000
    z = np.linspace(a_input, b_input, num_intervals_input + 1)

    # Calculate data length
    data_length = data.shape[0]

    # Repeat z values for each row in data
    z_repeated = np.tile(np.expand_dims(z, axis=0), [data_length, 1])

    # Define z0
    z0 = 0.0

    # Create z0_tensor
    z0_tensor = np.expand_dims(np.full(data_length, z0), axis=1)

    # Prepare inputs dictionary
    X = {
        "features_n": np.vstack(
            data[
                [
                    "geo_kz_ml",
                    "geo_thetainc_ml",
                    "geo_amp",
                    "geo_coh",
                    "geo_pha",
                ]
            ].values
        ),
        "z": z_repeated,
        "kappa_z": data["geo_kz_ml"].values[:, np.newaxis],
        "z0": z0_tensor,
        "kappa_z_vol": data["geo_kz_ml"].values[:, np.newaxis],
    }

    # Prepare optional output dictionary if output is provided
    y = {}
    if output:
        for key, value in output.items():
            if value in data.columns:
                y[key] = data[[value]].values
            else:
                print(
                    f"Warning: Column '{value}' not \
                        found in DataFrame for output '{key}'"
                )

    return X, y
