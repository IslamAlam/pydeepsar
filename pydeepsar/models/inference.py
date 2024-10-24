"""Perform inference on all NetCDF files in a given directory using a specified model."""

"""
This script performs inference on all NetCDF files in a given directory
using a specified model.

Example of how to call this script from the command line:

    python -m pydeepsar.models.inference --inference_model_path /path/to/model \
        --tandem_root /path/to/tandem/data

You can also provide optional parameters:

    python -m pydeepsar.models.inference --inference_model_path /path/to/model \
        --tandem_root /path/to/tandem/data --layer_names layer1 layer2 --reference_bias my_bias --y_reference my_y_reference --y_estimated my_y_estimated

    python -m pydeepsar.models.inference \
        --inference_model_path \
    "/dss/dsshome1/01/di93sif/di93sif/ice/pydeepsar/models/model-20240420-215546.275" \
    --tandem_root "/dss/dsshome1/01/di93sif/di93sif/Summit_East_Coast_2017" \
    --layer_names "d_pen,coherence,PhaseCenterDepth,phase" \
    --reference_bias "bias" --y_reference "iodem3_2017_DEM" \
    --y_estimated "SEC_and_cnst_offset_postp_dem"

    python -m pydeepsar.models.inference \
        --inference_model_path \
    "/dss/dsshome1/01/di93sif/di93sif/ice/pydeepsar/models/model-20240427-165512.185" \
    --tandem_root "/dss/dsshome1/01/di93sif/di93sif/Summit_East_Coast_2017" \
    --layer_names "d_pen,coherence,PhaseCenterDepth,phase" \
    --reference_bias "bias" --y_reference "iodem3_2017_DEM" \
    --y_estimated "SEC_and_cnst_offset_postp_dem"

    python -m pydeepsar.models.inference \
    --inference_model_path \
    "/dss/dsshome1/01/di93sif/di93sif/ice/pydeepsar/models/model-20240427-165512.185" \
    --tandem_root "/dss/dsshome1/01/di93sif/di93sif/Summit_East_Coast_2017" \
    --layer_names "d_pen,coherence,PhaseCenterDepth,phase" --reference_bias "bias" \
    --y_reference "iodem3_2017_DEM" --y_estimated "SEC_and_cnst_offset_postp_dem"
"""

# %%
from pathlib import Path
from typing import Dict, List, Optional

import click
import numpy as np
import tensorflow as tf

from tqdm import tqdm

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

from pydeepsar.io.xr import (
    GeoNetCDFDataReader,
    TandemXData,
    update_dataset_with_dataframe,
)
from pydeepsar.models.coherene import create_model_input_output

# %%
# from concurrent.futures import ProcessPoolExecutor

# inference_model_path="/dss/dsshome1/01/di93sif/di93sif/ice/pydeepsar/models/model-20240420-215546.275"
# tandem_root="/dss/dsshome1/01/di93sif/di93sif/Summit_East_Coast_2017"

# layer_names="d_pen,coherence,PhaseCenterDepth,phase"
# if layer_names is not None:
#     layer_names = layer_names.split(',')
# reference_bias="bias"
# y_reference="iodem3_2017_DEM"
# y_estimated="SEC_and_cnst_offset_postp_dem"

# tandem_data = TandemXData(tandem_root)
# tandem_netcdf_paths = tandem_data.get_netcdf_files()

# netcdf_file_path= tandem_netcdf_paths[0]


def do_inference_and_save(
    inference_model_path: Path,
    netcdf_file_path: Path,
    layer_names: Optional[Dict[str, Dict[str, float]]],
    reference_bias: str,
    y_reference: str,
    y_estimated: str,
) -> None:
    # Convert inference_model_path to a Path object
    inference_model_path = Path(inference_model_path)

    # Load the model
    models_path = Path(inference_model_path)
    models_paths = sorted(list(models_path.glob("model-*/")), reverse=True)
    models_paths = [f for f in models_paths if f.is_dir()]
    latest_model_path = models_paths[0]
    model = tf.keras.models.load_model(latest_model_path)

    # Construct the output file path
    output_file_path = netcdf_file_path.parent.joinpath(
        f"./ds_predict_{inference_model_path.name}.geo.nc"
    )

    # If the output file already exists, skip the inference for this file
    if output_file_path.exists():
        # print(f"Output file {output_file_path} already exists. Skipping inference for this file.")
        click.echo(
            f"Output file {output_file_path} already exists. Skipping inference for this file."
        )

    else:
        # with tf.device('/cpu:0'):
        # Read the netcdf_file_path file
        reader = GeoNetCDFDataReader(netcdf_file_path)
        ds = reader.read_netcdf()

        # Convert dataset to DataFrame
        dataframe = ds.to_dataframe()

        # Use default layer names if not provided
        if layer_names is None:
            layer_names = {
                "PhaseCenterDepth": {"vmin": -10, "vmax": 0},
                "d_pen": {"vmin": -30, "vmax": 0},
                "phase": {"vmin": -np.pi, "vmax": np.pi},
            }

        # Define input for prediction
        inputs, _ = create_model_input_output(dataframe=dataframe, output=None)

        layer_names_list = list(layer_names.keys())

        # Define a submodel that outputs the desired layers' outputs
        desired_layers_outputs = [
            model.get_layer(name=layer_name).output
            for layer_name in layer_names_list  # type: ignore[union-attr]
        ]
        desired_layers_model = tf.keras.Model(
            inputs=model.input, outputs=desired_layers_outputs
        )

        # Get the output values of the desired layers for the given input
        output_values = desired_layers_model.predict(inputs)

        # Add the output values of the desired layers to dataframe
        for layer_name, output_value in zip(layer_names_list, output_values):  # type: ignore[arg-type]
            dataframe[layer_name] = output_value
        # for layer_name, output_value in zip(layer_names, [output_values] if not isinstance(output_values, list) else output_values):  # type: ignore[arg-type]
        #     dataframe[layer_name] = output_value

        ds[reference_bias] = -(ds[y_reference] - ds[y_estimated])

        updated_ds = update_dataset_with_dataframe(
            ds, dataframe[layer_names_list]
        )

        # Save the updated dataset to a NetCDF file
        updated_ds.to_netcdf(output_file_path, engine="h5netcdf")

        click.echo(f"Inference completed and saved to {output_file_path}")


# Define a function to process a single file
def process_file(
    netcdf_file_path: Path,
    inference_model_path: Path,
    layer_names: Optional[Dict[str, Dict[str, float]]],
    reference_bias: str,
    y_reference: str,
    y_estimated: str,
) -> None:
    do_inference_and_save(
        inference_model_path,
        netcdf_file_path,
        layer_names,  # type: ignore[arg-type]
        reference_bias,
        y_reference,
        y_estimated,
    )


def parse_layer_names(layer_names_str):
    """
    Convert a comma-separated string of layer names into a dictionary with vmin and vmax set to None.

    Args:
        layer_names_str (str): Comma-separated names of the layers.

    Returns:
        Dict[str, Dict[str, Optional[float]]]: Dictionary with layer names as keys
        and vmin/vmax as sub-keys.
    """
    # Convert the input string to a list of layer names
    layer_names_list = layer_names_str.split(",")

    # Create the layer_names dictionary with vmin and vmax set to NaN
    layer_names = {
        name: {"vmin": None, "vmax": None} for name in layer_names_list
    }

    return layer_names


@click.command()
@click.option(
    "--inference_model_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the inference model.",
)
@click.option(
    "--tandem_root",
    required=True,
    type=click.Path(exists=True),
    help="Root directory of the TandemX data.",
)
@click.option(
    "--layer_names",
    default=None,
    help="Comma-separated names of the layers for inference.",
)
@click.option(
    "--reference_bias",
    default="reference_bias",
    help="Name of the reference bias.",
)
@click.option(
    "--y_reference", default="iodem3_2017_DEM", help="Name of the y reference."
)
@click.option(
    "--y_estimated",
    default="SEC_and_cnst_offset_postp_dem",
    help="Name of the y estimated.",
)
# @click.option(
#     "--num_jobs",
#     default=None,
#     type=int,
#     help="Number of parallel jobs to run. Defaults to the number of processors.",
# )
def process_all_tandem_data(
    inference_model_path: Path,
    tandem_root: Path,
    layer_names: Optional[str],
    reference_bias: str,
    y_reference: str,
    y_estimated: str,
) -> None:
    # Convert layer_names to a list
    if layer_names is not None:
        # layer_names = layer_names.split(",")  # type: ignore[assignment]
        layer_names = parse_layer_names(layer_names)

    # Get all the NetCDF file paths
    tandem_data = TandemXData(tandem_root)  # type: ignore[arg-type]
    tandem_netcdf_paths = tandem_data.get_netcdf_files()
    print(
        tandem_netcdf_paths,
        inference_model_path,
    )

    # Run process_file for each file in parallel
    # with ProcessPoolExecutor(max_workers=num_jobs) as executor:
    #     list(tqdm(executor.map(process_file, tandem_netcdf_paths, inference_model_path, layer_names, reference_bias, y_reference, y_estimated), total=len(tandem_netcdf_paths), desc="Processing files"))    # Run do_inference_and_save for each file
    for netcdf_file_path in tqdm(tandem_netcdf_paths, desc="Processing files"):
        do_inference_and_save(
            inference_model_path,
            netcdf_file_path,
            layer_names, # type: ignore[arg-type]
            reference_bias,
            y_reference,
            y_estimated,
        )


if __name__ == "__main__":
    process_all_tandem_data()
