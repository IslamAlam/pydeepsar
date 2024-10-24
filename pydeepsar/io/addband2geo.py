"""
Merge two NetCDF files into a single file.

The function reprojects the second file to match the projection of the
first file, and adds its reprojected bands to the first file.
"""

import argparse
import os
import re

from pathlib import Path
from typing import Union

import rioxarray as rxr
import xarray as xr

from rasterio.enums import Resampling
from shapely.geometry import box


def merge_nc_files(
    input_path1: Union[str, Path],
    input_path2: Union[str, Path],
    output_path: Union[str, Path],
    prefix: str = "band_",
    method: Resampling = Resampling.average,
) -> xr.Dataset:
    """
    Merge two NetCDF files into a single file.

    The function reprojects the second file to match the projection of the
    first file, and adds its reprojected bands to the first file.

    Args:
        input_path1 (str): Path to the first NetCDF file to merge.
        input_path2 (str): Path to the second NetCDF file to merge.
        output_path (str): Path to write the merged file to.
        prefix (str, optional): Prefix to use for the band names in the merged
        dataset. Defaults to "band_".
        method (rasterio.enums.Resampling, optional): Resampling method to use
        when reprojecting the second dataset. Defaults to Resampling.average.

    Returns:
    --------
        xr.Dataset: Merged NetCDF dataset.
    """
    # Load the input NetCDF files using rioxarray
    with rxr.open_rasterio(Path(input_path1), mask_and_scale=True) as data1:  # type: ignore
        with rxr.open_rasterio(
            Path(input_path2), mask_and_scale=True
        ) as data2:  # type: ignore
            # Reproject the second data to match the projection of the first data
            data1 = data1.load()

            # data2 = data2.load()
            data2 = data2

            # Check if the CRS is not empty
            if data2.rio.crs is not None:
                print("CRS is not empty:", data2.rio.crs)
            else:
                print("CRS is empty.")
                data2 = data2.rio.write_crs(4326, inplace=True)

            boundary_extent = data1.rio.bounds()
            # Convert the bounding box coordinates to a shapely box object
            boundary_box = box(*boundary_extent)

            # Check if the bounds of data2 intersect with the bounding box
            if box(*data2.rio.bounds()).intersects(boundary_box):
                reproj_data2 = data2.rio.clip_box(*boundary_extent)
                # Continue with further processing using the clipped data
                reproj_data2 = reproj_data2.rio.reproject_match(
                    data1, resampling=method
                )

            else:
                print("No data found within the specified bounding box.")
                # Handle the case where no data is present within the bounding box
                reproj_data2 = data2.rio.reproject_match(
                    data1, resampling=method
                )

            # Check if the input data is a Dataset or DataArray, and convert reproj_data2 to a Dataset if necessary
            if isinstance(data1, xr.Dataset):
                if isinstance(reproj_data2, xr.DataArray):
                    merged_data = data1
                    if len(reproj_data2.band) == 1:
                        data = reproj_data2[0, :, :]
                        band = data.band
                        data = data.expand_dims("band", axis=0)
                        data.coords["band"] = [1]
                        # merged_data = xr.merge([merged_data, data.to_dataset(name=prefix+str(int(band)))], )
                        merged_data[prefix] = (("band", "y", "x"), data.values)

                    else:
                        for i in range(len(reproj_data2.band)):
                            data = reproj_data2[i, :, :]
                            band = data.band
                            data = data.expand_dims("band", axis=0)
                            data.coords["band"] = [1]
                            # merged_data = xr.merge([merged_data, data.to_dataset(name=prefix+str(int(band)))], )
                            merged_data[prefix + str(int(band))] = (
                                ("band", "y", "x"),
                                data.values,
                            )

                elif isinstance(reproj_data2, xr.Dataset):
                    merged_data = xr.merge(
                        [
                            reproj_data2,
                            data1,
                        ],
                        compat="override",
                    )

    # Write CRS and transform information to the merged data
    merged_data.rio.write_crs(data1.rio.crs, inplace=True)
    merged_data.rio.write_transform(inplace=True)

    output_dir_path = Path(output_path).parent
    output_dir_path.mkdir(parents=True, exist_ok=True)
    # Save the merged data to a new NetCDF file
    merged_data.to_netcdf(Path(output_path), engine="h5netcdf")

    return merged_data


def is_valid_file(file_path: str) -> str:
    """
    Check whether the given file path exists or not.

    Args:
        file_path (str): Path to the file to check.

    Returns
    -------
        str: The path to the file if it exists.

    Raises
    ------
        argparse.ArgumentTypeError: If the file does not exist.
    """
    if not os.path.isfile(file_path):
        raise argparse.ArgumentTypeError(
            f"{file_path} does not exist or is not a file."
        )
    return file_path


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Merge two NetCDF files into a single file, by reprojecting the second file to match the projection of the first file, and adding its reprojected bands to the first file."
    )
    parser.add_argument(
        "--input_path1",
        type=is_valid_file,
        help="Path to the first NetCDF file to merge.",
    )
    parser.add_argument(
        "--input_path2",
        type=is_valid_file,
        help="Path to the second NetCDF file to merge.",
    )
    parser.add_argument(
        "--output_path", type=str, help="Path to write the merged file to."
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="band_",
        help="Prefix to use for the band names in the merged dataset. Defaults to 'band_'.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="average",
        choices=["nearest", "bilinear", "cubic", "average", "mode"],
        help="Resampling method to use when reprojecting the second dataset. Defaults to 'average'.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Map string value to rasterio Resampling enum
    method = Resampling[args.method]

    # Merge the NetCDF files and write the merged data to a new file
    merge_nc_files(
        args.input_path1,
        args.input_path2,
        args.output_path,
        args.prefix,
        method,
    )


if __name__ == "__main__":
    main()
