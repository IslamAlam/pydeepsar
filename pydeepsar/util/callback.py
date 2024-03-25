"""
The module provides several callbacks during training.

The module provides several TensorFlow callbacks for visualizing and logging
layer outputs during model training.


Classes:
    - PrintLayerValuesCallback: Callback to print the output values of \
        specific layers at the end of each epoch.
    - PlotLayerImageCallback: Callback to plot and save images of specific \
        layers at the end of each epoch.
    - LogFiguresCallback: Callback to log figures of specific layers at \
        the end of each epoch.

Functions:
    - plot_to_image: Convert a matplotlib figure to a TensorFlow image tensor.

Dependencies:
    - io
    - Any
    - List
    - matplotlib.pyplot
    - numpy
    - pandas
    - tensorflow
    - matplotlib.figure.Figure

Example:
    ```python
    from datetime import datetime
    import io

    import tensorflow as tf
    from tensorflow import keras

    import matplotlib.pyplot as plt
    import numpy as np

    logdir = "logs/image/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # Define the basic TensorBoard callback.
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    file_writer_figures = tf.summary.create_file_writer(logdir + "/figures")
    callback = LogFiguresCallback(dataset, file_writer_figures)
    model.fit(x_train, y_train, epochs=10, callbacks=[callback])
    ```
"""

import io

from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from matplotlib.figure import Figure

from pydeepsar.io.xr import update_dataset_with_dataframe
from pydeepsar.models.coherene import create_model_input_output


class PrintLayerValuesCallback(tf.keras.callbacks.Callback):  # type: ignore[misc]
    """Callback to print layer outputs at the end of epochs.

    This callback prints the output values of specific layers of the model \
        at the end of each epoch during training.

    Parameters
    ----------
    inputs : tf.Tensor
        The input data to be used for obtaining layer outputs.
    layer_names : List[str], optional
        List of names of the layers whose values are to be printed.
        If not provided, default layer names will be used.

    Examples
    --------
    >>> inputs = tf.constant(...)  # Provide input data
    >>> print_callback = PrintLayerValuesCallback(inputs)
    >>> model.fit(x_train, y_train, epochs=10, callbacks=[print_callback])
    """

    def __init__(
        self, inputs: tf.Tensor, layer_names: Optional[List[str]] = None
    ) -> None:
        """Initialize the PrintLayerValuesCallback.

        Parameters
        ----------
        inputs : tf.Tensor
            The input data to be used for obtaining layer outputs.
        layer_names : List[str], optional
            List of names of the layers whose values are to be printed.
            If not provided, default layer names will be used.
        """
        self.inputs = inputs
        if layer_names is None:
            self.layer_names = [
                "d_pen_prediction",
                "ComplexCoherence",
                "coherence",
                "PhaseCenterDepth",
                "phase",
            ]
        else:
            self.layer_names = layer_names

    def on_epoch_end(self, epoch: int, **kwargs: Any) -> None:
        """Print the output values of specific layers at the end of each epoch.

        This method prints the output values of specific layers of the model
        at the end of each epoch during training.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        logs : dict or None
            Dictionary containing the training metrics for the current epoch \
                (optional).
        """
        print(f"Epoch {epoch+1} - Layer Values:")

        # Define a submodel that outputs the desired layers' outputs
        desired_layers_outputs = [
            self.model.get_layer(name=layer_name).output
            for layer_name in self.layer_names
        ]
        desired_layers_model = tf.keras.Model(
            inputs=self.model.input, outputs=desired_layers_outputs
        )

        # Get the output values of the desired layers for the given input
        output_values = desired_layers_model.predict(self.inputs)

        # Print the output values of the desired layers
        for layer_name, output_value in zip(self.layer_names, output_values):
            print(f"Output value of layer {layer_name}: {output_value}")


class PlotLayerImageCallback(tf.keras.callbacks.Callback):  # type: ignore[misc]
    """Callback to save images of specific layers at the end of each epoch.

    This callback plots and saves images of specific layers of the model
    at the end of each epoch during training.

    Parameters
    ----------
    dataset : tf.data.Dataset
        The dataset used for plotting layer images.
    save_path : str
        The directory path where the images will be saved.
    layer_names : list of str, optional
        List of names of the layers whose values are to be plotted and saved.
        If not provided, default layer names will be used.

    Examples
    --------
    >>> callback = PlotLayerImageCallback(dataset, save_path='./layer_images')
    >>> model.fit(x_train, y_train, epochs=10, callbacks=[callback])
    """

    def __init__(
        self,
        dataset: tf.data.Dataset,
        save_path: str,
        layer_names: Optional[List[str]] = None,
    ) -> None:
        """Initialize the PlotLayerImageCallback.

        Parameters
        ----------
        dataset : tf.data.Dataset
            The dataset used for plotting layer images.
        save_path : str
            The directory path where the images will be saved.
        layer_names : list of str, optional
            List of names of the layers whose values are to be plotted \
                and saved.
            If not provided, default layer names will be used.
        """
        self.dataset = dataset
        self.save_path = str(save_path)
        if layer_names is None:
            self.layer_names = [
                "d_pen_prediction",
                "ComplexCoherence",
                "coherence",
                "PhaseCenterDepth",
                "phase",
            ]
        else:
            self.layer_names = layer_names

    def on_epoch_end(self, epoch: int, **kwargs: Any) -> None:
        """Plot and save images of specific layers at the end of each epoch.

        This method plots and saves images of specific layers of the model
        at the end of each epoch during training.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        logs : dict or None
            Dictionary containing the training metrics for the current epoch \
                (optional).
        """
        print(f"Epoch {epoch+1} - Layer Images:")

        # Convert dataset to DataFrame
        dataframe = self.dataset.to_dataframe()

        # Define input for prediction
        self.inputs, _ = create_model_input_output(
            dataframe=dataframe, output=None
        )

        # Define a list of desired layers' names
        layer_names = self.layer_names

        # Define a submodel that outputs the desired layers' outputs
        desired_layers_outputs = [
            self.model.get_layer(name=layer_name).output
            for layer_name in layer_names
        ]
        desired_layers_model = tf.keras.Model(
            inputs=self.model.input, outputs=desired_layers_outputs
        )

        # Get the output values of the desired layers for the given input
        output_values = desired_layers_model.predict(self.inputs)

        # Add the output values of the desired layers to dataframe
        for layer_name, output_value in zip(layer_names, output_values):
            # print(f"Output value of layer {layer_name}: {output_value}")
            dataframe[layer_name] = output_value

        updated_ds = update_dataset_with_dataframe(
            self.dataset, dataframe[layer_names]
        )

        for i, layer_name in enumerate(layer_names):
            fig, ax = plt.subplots(figsize=(10, 10))

            # Plot absolute value if dtype is complex
            if np.iscomplexobj(updated_ds[layer_name]):
                updated_ds[layer_name].squeeze().plot.imshow(ax=ax)

            ax.set_title(f"Output value of layer {layer_name}")
            plt.savefig(
                f"{self.save_path}/epoch_{epoch+1}_layer_{layer_name}.png"
            )
            plt.close(fig)


class LogFiguresCallback(tf.keras.callbacks.Callback):  # type: ignore[misc]
    """Callback to log figures of specific layers at the end of each epoch.

    This callback logs figures of specific layers of the model
    at the end of each epoch during training.

    Parameters
    ----------
    dataset : tf.data.Dataset
        The dataset used for logging layer figures.
    file_writer_figures : tf.summary.FileWriter
        The FileWriter object to write the figures as image summaries.
    layer_names : list of str, optional
        List of names of the layers whose figures are to be logged.
        If not provided, default layer names will be used.

    Examples
    --------
    >>>  from datetime import datetime
    >>>  import io

    >>>  import tensorflow as tf
    >>>  from tensorflow import keras

    >>>  import matplotlib.pyplot as plt
    >>>  import numpy as np

    >>> logdir = "logs/image/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    >>> # Define the basic TensorBoard callback.
    >>> tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    >>> file_writer_figures = tf.summary.create_file_writer(logdir + '/fig')
    >>> callback = LogFiguresCallback(dataset, file_writer_figures)
    >>> model.fit(x_train, y_train, epochs=10, callbacks=[callback])
    """

    def __init__(
        self,
        dataset: tf.data.Dataset,
        file_writer_figures: tf.summary.FileWriter,
        layer_names: Optional[List[str]] = None,
    ) -> None:
        """Initialize the LogFiguresCallback.

        Parameters
        ----------
        dataset : tf.data.Dataset
            The dataset used for logging layer figures.
        file_writer_figures : tf.summary.FileWriter
            The FileWriter object to write the figures as image summaries.
        layer_names : list of str, optional
            List of names of the layers whose figures are to be logged.
            If not provided, default layer names will be used.
        """
        super(LogFiguresCallback, self).__init__()
        self.dataset = dataset
        self.file_writer_figures = file_writer_figures
        if layer_names is None:
            self.layer_names = [
                "d_pen_prediction",
                "coherence",
                "PhaseCenterDepth",
                "phase",
            ]
        else:
            self.layer_names = layer_names

    def on_epoch_end(self, epoch: int, **kwargs: Any) -> None:
        """Log figures of specific layers at the end of each epoch.

        This method logs figures of specific layers of the model
        at the end of each epoch during training.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        logs : dict or None
            Dictionary containing the training metrics for the current epoch \
                (optional).
        """
        # Convert dataset to DataFrame
        dataframe = self.dataset.to_dataframe()

        # Define input for prediction
        inputs, _ = create_model_input_output(dataframe=dataframe, output=None)

        # Define a list of desired layers' names
        layer_names = self.layer_names

        # Define a submodel that outputs the desired layers' outputs
        desired_layers_outputs = [
            self.model.get_layer(name=layer_name).output
            for layer_name in layer_names
        ]
        desired_layers_model = tf.keras.Model(
            inputs=self.model.input, outputs=desired_layers_outputs
        )

        # Get the output values of the desired layers for the given input
        output_values = desired_layers_model.predict(inputs)

        # Add the output values of the desired layers to dataframe
        for layer_name, output_value in zip(layer_names, output_values):
            dataframe[layer_name] = output_value

        updated_ds = update_dataset_with_dataframe(
            self.dataset, dataframe[layer_names]
        )

        fig_image = {}
        for i, layer_name in enumerate(layer_names):
            fig, ax = plt.subplots(figsize=(6, 6))

            # Plot absolute value if dtype is complex
            if not np.iscomplexobj(updated_ds[layer_name]):
                updated_ds[layer_name].squeeze().plot.imshow(ax=ax)

            ax.set_title(f"Output value of layer {layer_name}")

            fig_image[layer_name] = plot_to_image(fig)

        # Log the figures as image summaries.
        with self.file_writer_figures.as_default():
            for key, value in fig_image.items():
                tf.summary.image(key, value, step=epoch)


def plot_to_image(figure: Figure) -> tf.Tensor:
    """Convert the matplotlib plot to tensor.

    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it as a TensorFlow image tensor.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The matplotlib figure object to convert.

    Returns
    -------
    tf.Tensor
        A TensorFlow image tensor representing the converted PNG image.
    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
