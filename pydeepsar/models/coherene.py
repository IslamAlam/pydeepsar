"""
Module for computing coherence measures using TensorFlow.

This module provides functions for computing coherence measures
using TensorFlow, a popular deep learning framework.


"""

# %%

from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from pydeepsar.models.layers import (
    ComplexCoherenceEstimatorLayer,
    UniformVolumeLayer,
)


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


class CoherenceIceModel:
    """Class for creating a model for coherence estimation in ice.

    This class creates a TensorFlow model for estimating coherence in \
        ice based on the provided inputs.

    Attributes
    ----------
    d_pen_input : tf.Tensor
        The input tensor for the parameter d_pen.
    output_Profile : tf.Tensor
        The output tensor from the UniformVolumeLayer.
    model_UV : tf.keras.Model
        The TensorFlow model for the uniform volume layer.
    kappa_z_input : tf.Tensor
        The input tensor for the parameter kappa_z.
    kappa_z_vol_input : tf.Tensor
        The input tensor for the parameter kappa_z_vol.
    z0_input : tf.Tensor
        The input tensor for the parameter z0.
    combined_input : List[tf.Tensor]
        The combined input tensors for the coherence model.
    inputsCohModel : List[tf.Tensor]
        The input tensors for the coherence model.
    coh_est : tf.Tensor
        The output tensor from the ComplexCoherenceEstimatorLayer.
    model : tf.keras.Model
        The complete TensorFlow model for coherence estimation in ice.

    Methods
    -------
    create_UV_model()
        Create the model for the uniform volume layer.
    create_coherence_model()
        Create the model for coherence estimation.
    plot_model()
        Plot the architecture of the coherence model.

    Examples
    --------
    # Create an instance of CoherenceIceModel
    ice_model = CoherenceIceModel()

    # Plot the architecture of the coherence model
    ice_model.plot_model()
    """

    def __init__(self) -> None:
        self.d_pen_input: Optional[tf.Tensor] = None
        self.output_Profile: Optional[tf.Tensor] = None
        self.model_UV: Optional[tf.keras.Model] = None
        self.kappa_z_input: Optional[tf.Tensor] = None
        self.kappa_z_vol_input: Optional[tf.Tensor] = None
        self.z0_input: Optional[tf.Tensor] = None
        self.combined_input: Optional[tf.Tensor] = None
        self.inputsCohModel: Optional[tf.Tensor] = None
        self.coh_est: Optional[tf.Tensor] = None
        self.model: Optional[tf.keras.Model] = None

        self.create_UV_model()
        self.create_coherence_model()

    def create_UV_model(self) -> None:
        """Create the model for the uniform volume layer."""
        self.d_pen_input = tf.keras.Input(shape=(1,), name="d_pen")
        z_input = tf.keras.Input(shape=(None,), name="z")
        self.output_Profile = UniformVolumeLayer(
            name="UniformVolumeModelProfile"
        )([z_input, self.d_pen_input])
        self.model_UV = tf.keras.Model(
            inputs=[z_input, self.d_pen_input], outputs=self.output_Profile
        )

    def create_coherence_model(self) -> tf.keras.Model:
        """Create the model for coherence estimation."""
        self.kappa_z_input = tf.keras.Input(shape=(1,), name="kappa_z")
        self.kappa_z_vol_input = tf.keras.Input(shape=(1,), name="kappa_z_vol")
        self.z0_input = tf.keras.Input(shape=(1,), name="z0")

        self.combined_input = [
            self.model_UV.input[0],  # type: ignore[union-attr]
            self.d_pen_input,
            self.kappa_z_input,
            self.kappa_z_vol_input,
            self.z0_input,
        ]
        self.inputsCohModel = [
            self.model_UV.input[0],  # type: ignore[union-attr]
            self.output_Profile,
            self.kappa_z_input,
            self.kappa_z_vol_input,
            self.z0_input,
        ]

        self.coh_est = ComplexCoherenceEstimatorLayer(name="ComplexCoherence")(
            self.inputsCohModel
        )

        self.model = tf.keras.Model(
            inputs=self.combined_input, outputs=self.coh_est
        )

        return self.model

    def plot_model(self) -> None:
        """Plot the architecture of the coherence model."""
        tf.keras.utils.plot_model(
            self.model,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=300,
            show_layer_activations=True,
            show_trainable=True,
        )


# %%
