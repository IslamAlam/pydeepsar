"""Provides functions for plotting data."""

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator


def plot_x_y(
    x: npt.NDArray[Union[np.float32, np.float64]],
    y: npt.NDArray[Union[np.float32, np.float64]],
    label: Optional[str] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plot the relationship between x and y values.

    Parameters
    ----------
    x : array-like
        Array of x values.
    y : array-like
        Array of y values.
    label : str, optional
        Label for the plotted line.
    ax : matplotlib.axes.Axes, optional
        The axis to plot on. If None, a new figure and axis will be created.

    Returns
    -------
    matplotlib.axes.Axes
        The axis object.
    """
    if ax is None:
        # Create a new figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))

    # Plot x against y
    ax.plot(x, y, label=label)

    # Set minor ticks for x-axis
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))

    # Set minor ticks for y-axis
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    # Set x and y limits
    ax.set_xlim(0, 0.6)
    ax.set_ylim(-25, 0)

    # Add a legend
    ax.legend()

    # Add grid lines
    ax.grid(True, which="major")

    # Return the axis
    return ax


def plot_complex_polar(
    z: npt.NDArray[Union[np.complex64, np.complex128]],
    label: Optional[str] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plot a complex number on a polar plot.

    Parameters
    ----------
    complex_number : complex
        The complex number to be plotted.
    label : str, optional
        The label for the plotted point. Default is None.
    ax : matplotlib.axes.Axes, optional
        The axis to plot on. If None, a new figure and axis will be created.

    Returns
    -------
    matplotlib.axes.Axes
        The axis object.
    """
    # Calculate magnitude and phase angle
    magnitude = np.abs(z)
    phase = np.angle(z, deg=True)

    if ax is None:
        # Create a new figure and axis
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, polar=True)

    # Plot the complex number on the polar plot
    ax.scatter([phase], [magnitude], label=label)

    # Set radius limits
    ax.set_ylim(0, 1)

    # Add a grid
    ax.grid(True)

    # Add a legend
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    # Return the axis
    return ax
