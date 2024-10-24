# %%
import re

from pathlib import Path

import numpy as np
import rioxarray as rxr
import xarray as xr

from rioxarray.merge import merge_datasets


class DatFileReader:
    def __init__(self, folder_path):
        self.folder_path = Path(folder_path)

    def list_dat_files(self):
        dat_files = list(self.folder_path.glob("**/L2-PRI/*.dat"))
        return dat_files

    def read_dat_file(self, file_path):
        header_dt = np.dtype(np.float64)
        data_dt = np.dtype(np.float32)

        # header = np.fromfile(file_path, dtype=header_dt, count=6).byteswap()
        # taxi_dem = np.fromfile(
        #     file_path, dtype=data_dt, offset=int(64 / 8 * 6)
        # ).byteswap()

        # taxi_dem = np.flipud(taxi_dem.reshape(int(header[1]), int(header[0])))

        header = np.fromfile(file_path, dtype=header_dt, count=6).byteswap()
        taxi_dem = np.fromfile(
            file_path, dtype=data_dt, offset=int(64 / 8 * 6)
        ).byteswap()

        # taxi_dem = np.flipud(taxi_dem.reshape(int(header[1]), int(header[0])))
        # print(file_path.name)
        taxi_dem = taxi_dem.reshape(int(header[1]), int(header[0]))

        taxi_dem[taxi_dem == -9999] = np.nan

        lon = header[[2, 4]]
        lat = header[[3, 5]]
        postlon = abs(header[4] - header[2]) / (header[0] - 1)
        postlat = abs(header[5] - header[3]) / (header[1] - 1)

        coords = {
            "x": np.linspace(lon[0], lon[1], int(header[0])),
            "y": np.linspace(lat[0], lat[1], int(header[1])),
        }

        data_array = xr.DataArray(taxi_dem, coords=coords, dims=["y", "x"])
        data_array.rio.write_crs(4326, inplace=True)
        data_array.rio.set_crs(4326, inplace=True)
        data_array.rio.set_nodata(np.nan)
        return data_array

    def read_and_create_dataset(
        self,
    ):
        dat_files = self.list_dat_files()
        datasets = []

        for file in dat_files:
            data_array = self.read_dat_file(file)
            data_array.rio.write_crs(
                4326, inplace=True
            )  # Set the coordinate reference system
            var_name = extract_filename(file.name)
            dataset = data_array.to_dataset(name=var_name)
            datasets.append(dataset)

        if datasets:
            combined_dataset = xr.merge(datasets)
            combined_dataset = combined_dataset.rio.set_crs(4326, inplace=True)
            combined_dataset = combined_dataset.rio.write_crs(
                4326, inplace=True
            )  # Set the coordinate reference system

            for var in combined_dataset.data_vars:
                combined_dataset[var].rio.set_crs(4326, inplace=True)
                combined_dataset[var].rio.write_crs(4326, inplace=True)

            combined_dataset.rio.write_crs(
                4326,
                inplace=True,
            ).rio.set_spatial_dims(
                x_dim="x",
                y_dim="y",
                inplace=True,
            ).rio.write_coordinate_system(
                inplace=True
            )

            return combined_dataset
        else:
            return None


def extract_filename(filename):
    match = re.search(r"^(.*?)(?=_m2017|\.dat)", filename)
    if match:
        return match.group(1)
    return None


# %%
if __name__ == "__main__":
    folder_path = r"./Summit_East_Coast_2017/TanDEM-X"
    scene_folders = sorted(list(Path(folder_path).glob("TDM*/")))
    write_path = Path(
        "/data/HR_Data/Pol-InSAR_InfoRetrieval/10_users/mans_is/DeepSAR-ICE/Summit_East_Coast_2017/TanDEM-X"
    )
    for scene_folder in scene_folders:
        reader = DatFileReader(scene_folder)
        xds = reader.read_and_create_dataset()

        filepath = write_path.joinpath(
            scene_folder.name, scene_folder.name + ".nc"
        )
        filepath.parent.mkdir(parents=True, exist_ok=True)
        xds.to_netcdf(filepath)

# %%
