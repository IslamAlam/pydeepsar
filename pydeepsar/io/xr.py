"""
Module for managing TanDEM-X and geospatial data.

This module provides classes and functions for managing TanDEM-X data files
and reading geospatial data from NetCDF files.

Classes:
- TandemXData: Class for managing TanDEM-X data files.
- GeoNetCDFDataReader: Class for reading geospatial data from NetCDF files.

Functions:
- update_dataset_with_dataframe: Update the original dataset with new variables
from a DataFrame.
- slice_middle_portion: Slice the middle portion of an xarray dataset
along both x and y dimensions.

Examples
--------
Importing the module:

>>> import tandemx_geodata as tg

Initializing TandemXData and accessing DEM files:

>>> tandem_root = "../../../../Summit_East_Coast_2017/TanDEM-X"
>>> tandem_data = tg.TandemXData(tandem_root)
>>> dem_files = tandem_data.get_dem_files()
>>> print(dem_files)
[Path('/path/to/dem_file1.nc'), Path('/path/to/dem_file2.nc'), ...]

Initializing GeoNetCDFDataReader and reading geospatial data:

>>> netcdf_file_path = "/path/to/netcdf_file.nc"
>>> reader = tg.GeoNetCDFDataReader(netcdf_file_path)
>>> ds = reader.read_netcdf()
>>> print(ds)
<xarray.Dataset>
Dimensions:     (lat: 100, lon: 100)
Coordinates:
  * lon         (lon) float32 -180.0 -179.1 -178.2 -177.3 ... -81.9 -81.0 -80.1
  * lat         (lat) float32 90.0 89.1 88.2 87.3 ... 1.9 1.0 0.1 -0.8
    ...

Updating dataset with new variables from a DataFrame:

>>> import xarray as xr
>>> import pandas as pd
>>> import numpy as np
>>> # Create a sample xarray dataset
>>> ds = xr.Dataset(
...     {
...         'temperature': (('time', 'location'), np.random.rand(5, 3)),
...         'pressure': (('time', 'location'), np.random.rand(5, 3)),
...     },
...     coords={'time': pd.date_range('2022-01-01', periods=5),
...     'location': ['A', 'B', 'C']}
... )
>>> # Create a sample DataFrame with new variables
>>> df = pd.DataFrame(
...     {
...         'humidity': [0.4, 0.5, 0.6, 0.7, 0.8],
...         'wind_speed': [10, 12, 15, 11, 9]
...     },
...     index=pd.date_range('2022-01-01', periods=5)
... )
>>> # Update the dataset with the new variables from the DataFrame
>>> updated_ds = tg.update_dataset_with_dataframe(ds, df)
>>> print(updated_ds)
<xarray.Dataset>
Dimensions:     (location: 3, time: 5)
Coordinates:
  * time        (time) datetime64[ns] 2022-01-01 2022-01-02 ... 2022-01-05
  * location    (location) object 'A' 'B' 'C'
Data variables:
    temperature  (time, location) float64 0.6453 ... 0.8361 0.6811
    pressure     (time, location) float64 0.9784 ... 0.6114 0.6046
    humidity     (time) float64 0.4 0.5 0.6 0.7 0.8
    wind_speed   (time) int64 10 12 15 11 9

Slicing the middle portion of an xarray dataset:

>>> # Create a sample xarray dataset
>>> x = np.linspace(0, 10, 100)
>>> y = np.linspace(0, 5, 50)
>>> data = np.random.rand(100, 50)
>>> ds = xr.Dataset(
...     {"data": (("x", "y"), data)},
...     coords={"x": x, "y": y}
... )
>>> # Slice the middle portion with 5 pixels on each side
>>> middle_ds = tg.slice_middle_portion(ds, 5)
>>> print(middle_ds)
<xarray.Dataset>
Dimensions:  (x: 10, y: 10)
Coordinates:
  * x        (x) float64 2.778 3.333 3.889 4.444 ... 6.667 7.222 7.778 8.333
  * y        (y) float64 0.278 0.8333 1.389 1.944 ... 3.889 4.444 5.0
Data variables:
    data     (x, y) float64 ...

"""

from pathlib import Path
from typing import List, Optional

import pandas as pd
import xarray as xr


class TandemXData:
    """Class for managing TanDEM-X data files.

    This class provides functionality to locate and manage TanDEM-X data files
    within a specified root directory.

    Parameters
    ----------
    root_dir : str
        The root directory where TanDEM-X data files are located.

    Attributes
    ----------
    root_dir : pathlib.Path
        The root directory where TanDEM-X data files are located.
    dem_files : List[pathlib.Path]
        A list of paths to TanDEM-X DEM files.


    Examples
    --------
    >>> tandem_root = "../../../../Summit_East_Coast_2017/TanDEM-X"
    >>> tandem_data = TandemXData(tandem_root)
    >>> dem_files = tandem_data.get_dem_files()
    >>> print(dem_files)
    [Path('/path/to/dem_file1.nc'), Path('/path/to/dem_file2.nc'), ...]
    """

    def __init__(self, root_dir: str) -> None:
        """Initialize TandemXData with the root directory."""
        self.root_dir = Path(root_dir)
        self.dem_files = sorted(self.root_dir.glob("**/TDM1_SAR__COS*.nc"))

    def get_dem_files(self) -> List[Path]:
        """Get a list of DEM files."""
        return self.dem_files


class GeoNetCDFDataReader:
    """Class for reading geospatial data from NetCDF files.

    This class provides functionality to read geospatial data
    from NetCDF files.

    Parameters
    ----------
    netcdf_file : pathlib.Path
        The path to the NetCDF file to be read.

    Attributes
    ----------
    netcdf_file : pathlib.Path
        The path to the NetCDF file.

    Examples
    --------
    >>> netcdf_file_path = "/path/to/netcdf_file.nc"
    >>> reader = GeoNetCDFDataReader(netcdf_file_path)
    >>> ds = reader.read_netcdf()
    >>> print(ds)
    <xarray.Dataset>
    Dimensions:     (lat: 100, lon: 100)
    Coordinates:
      * lon         (lon) float32 -180.0 ... -81.9 -81.0 -80.1
      * lat         (lat) float32 90.0 89.1 ... 1.9 1.0 0.1 -0.8
        ...
    """

    def __init__(self, netcdf_file: Path) -> None:
        """Initialize GeoNetCDFDataReader with the path to the NetCDF file."""
        self.netcdf_file = netcdf_file

    def read_netcdf(self) -> xr.Dataset:
        """Read geospatial data from the NetCDF file."""
        ds = xr.open_dataset(self.netcdf_file, decode_coords="all")
        return ds


def update_dataset_with_dataframe(
    ds: xr.Dataset, dataframe: pd.DataFrame
) -> xr.Dataset:
    """
    Update the original dataset with new variables from the DataFrame.

    Parameters
    ----------
    ds : xarray.Dataset
        The original dataset to be updated.
    dataframe : pandas.DataFrame
        The DataFrame containing new variables.

    Returns
    -------
    xarray.Dataset
        The updated dataset.

    Examples
    --------
    >>> import xarray as xr
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create a sample xarray dataset
    >>> ds = xr.Dataset(
    ...     {
    ...         'temperature': (('time', 'location'), np.random.rand(5, 3)),
    ...         'pressure': (('time', 'location'), np.random.rand(5, 3)),
    ...     },
    ...     coords={'time': pd.date_range('2022-01-01', periods=5),
    ...    'location': ['A', 'B', 'C']}
    ... )
    >>> # Create a sample DataFrame with new variables
    >>> df = pd.DataFrame(
    ...     {
    ...         'humidity': [0.4, 0.5, 0.6, 0.7, 0.8],
    ...         'wind_speed': [10, 12, 15, 11, 9]
    ...     },
    ...     index=pd.date_range('2022-01-01', periods=5)
    ... )
    >>> # Update the dataset with the new variables from the DataFrame
    >>> updated_ds = update_dataset_with_dataframe(ds, df)
    >>> print(updated_ds)
    <xarray.Dataset>
    Dimensions:     (location: 3, time: 5)
    Coordinates:
      * time        (time) datetime64[ns] 2022-01-01 2022-01-02 ... 2022-01-05
      * location    (location) object 'A' 'B' 'C'
    Data variables:
        temperature  (time, location) float64 0.6453 ... 0.8361 0.6811
        pressure     (time, location) float64 0.9784 ... 0.6114 0.6046
        humidity     (time) float64 0.4 0.5 0.6 0.7 0.8
        wind_speed   (time) int64 10 12 15 11 9
    """
    # Convert DataFrame to xarray Dataset
    dataset = dataframe.to_xarray()

    # Merge the original dataset with the new variables
    updated_ds = xr.merge([ds, dataset])

    return updated_ds


def slice_middle_portion(
    dataset: xr.Dataset,
    num_pixels: int,
    middle_x_index: Optional[int] = None,
    middle_y_index: Optional[int] = None,
) -> xr.Dataset:
    """
    Slice a portion of an xarray dataset along both x and y dimensions.

    Parameters
    ----------
    dataset : xarray.Dataset
        The xarray dataset to be sliced.
    num_pixels : int
        The number of pixels to include on each side of the middle
        along both x and y dimensions.
    middle_x_index : int, optional
        The middle index along the x dimension. If not provided,
        it will be calculated as the middle of the x dimension.
    middle_y_index : int, optional
        The middle index along the y dimension. If not provided,
        it will be calculated as the middle of the y dimension.

    Returns
    -------
    xarray.Dataset
        The sliced dataset containing the middle portion.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> # Create a sample xarray dataset
    >>> x = np.linspace(0, 10, 100)
    >>> y = np.linspace(0, 5, 50)
    >>> data = np.random.rand(100, 50)
    >>> ds = xr.Dataset(
    ...     {"data": (("x", "y"), data)},
    ...     coords={"x": x, "y": y}
    ... )
    >>> # Slice the middle portion with 5 pixels on each side
    >>> middle_ds = slice_middle_portion(ds, 5)
    >>> print(middle_ds)
    <xarray.Dataset>
    Dimensions:  (x: 10, y: 10)
    Coordinates:
      * x        (x) float64 2.778 3.333 ... 6.667 7.222 7.778 8.333
      * y        (y) float64 0.278 0.8333 ... 3.889 4.444 5.0
    Data variables:
        data     (x, y) float64 ...
    """
    # Calculate the middle indices along both x and y dimensions
    # if not provided
    if middle_x_index is None:
        middle_x_index = len(dataset["x"]) // 2
    if middle_y_index is None:
        middle_y_index = len(dataset["y"]) // 2

    # Define the range of indices for the middle portion along
    # both x and y dimensions
    start_x_index = max(0, middle_x_index - num_pixels)
    end_x_index = min(len(dataset["x"]), middle_x_index + num_pixels)
    start_y_index = max(0, middle_y_index - num_pixels)
    end_y_index = min(len(dataset["y"]), middle_y_index + num_pixels)

    # Slice the dataset along both x and y dimensions
    middle_dataset = dataset.isel(
        x=slice(start_x_index, end_x_index),
        y=slice(start_y_index, end_y_index),
    )

    return middle_dataset
