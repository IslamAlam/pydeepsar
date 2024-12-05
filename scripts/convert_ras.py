"""
Sun Raster (.ras) to GeoTIFF Converter

This module provides functionality to read Sun Raster (.ras) files and convert them
to GeoTIFF format. It handles various raster types including complex data, DEMs,
and SAR data with appropriate metadata preservation.

Supported raster types:
    - 89-90: Complex 4-bit integers
    - 91-92: 8/16-bit integers
    - 93-94: Complex 8-bit integers
    - 95-97: Complex float/double
    - 98-99: Float/double
    - 100: DEM data with footer
    - 101: SAR data with footer
    - 198-200: Geocoded data with footer

Usage:
    python convert_ras.py input.ras output.tiff

Author: Islam Mansour
Date: December 05, 2024
"""

# %%
import struct
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from osgeo import gdal, osr


@dataclass
class RasterFooter:
    """
    Data class representing the footer metadata of a Sun Raster file.

    Attributes:
        map: Map projection (countryID, projID, zonecode, datumcode)
        pixel_unit: Unit of pixel values (e.g., 'DE' for DEM)
        ul_northing: Upper-left corner northing coordinate
        ul_easting: Upper-left corner easting coordinate
        lr_northing: Lower-right corner northing coordinate
        lr_easting: Lower-right corner easting coordinate
        max_value: Maximum pixel value
        min_value: Minimum pixel value
        mean_value: Mean pixel value
        stdev: Standard deviation of pixel values
        lat_posting: Latitude pixel spacing
        lon_posting: Longitude pixel spacing
        inval: Invalid/NoData value
        inval_num: Number of invalid pixels
        ul_lat: Upper-left latitude
        ul_lon: Upper-left longitude
        ur_lat: Upper-right latitude
        ur_lon: Upper-right longitude
        ll_lat: Lower-left latitude
        ll_lon: Lower-left longitude
        lr_lat: Lower-right latitude
        lr_lon: Lower-right longitude
        res_lat: Latitude resolution
        res_lon: Longitude resolution
        dem_db_name: Optional database name for DEM files
    """

    map: str
    pixel_unit: str
    ul_northing: float
    ul_easting: float
    lr_northing: float
    lr_easting: float
    max_value: float
    min_value: float
    mean_value: float
    stdev: float
    lat_posting: float
    lon_posting: float
    inval: float
    inval_num: int
    ul_lat: float
    ul_lon: float
    ur_lat: float
    ur_lon: float
    ll_lat: float
    ll_lon: float
    lr_lat: float
    lr_lon: float
    res_lat: float
    res_lon: float
    dem_db_name: Optional[str] = None


def get_dtype(type: int, depth: int) -> np.dtype:
    """
    Determine numpy dtype based on raster type and bit depth.

    Args:
        type: Raster file type code (1-200)
        depth: Bit depth of the raster data

    Returns:
        np.dtype: Appropriate numpy dtype for the data

    Raises:
        ValueError: If type or depth combination is invalid
    """
    n_bytes = depth // 8

    if type == 1:  # Standard type
        if n_bytes == 1:
            return np.uint8
        elif n_bytes == 2:
            return np.int16
        elif n_bytes == 4:
            return np.int32
        elif n_bytes == 3:
            return np.uint8  # RGB triple
        elif n_bytes == 8:
            return np.complex64
        else:
            raise ValueError(f"Invalid bit depth {depth} for type 1")

    type_to_dtype = {
        89: np.uint8,  # 4-bit complex signed (needs post-processing)
        90: np.uint8,  # 4-bit complex unsigned (needs post-processing)
        91: np.uint16,  # unsigned 2-byte int
        92: np.int8,  # signed byte
        93: np.int8,  # complex signed bytes (needs post-processing)
        94: np.uint8,  # complex unsigned bytes (needs post-processing)
        95: np.complex128,  # double complex
        96: np.complex64,  # float complex
        97: np.complex64 if depth == 64 else np.int16,  # complex shorts or float
        98: np.float64,  # double
        99: np.float32,  # float
        100: {1: np.int8, 2: np.int16, 4: np.int32}.get(n_bytes, None),
        101: {1: np.uint8, 2: np.uint16, 4: np.uint32}.get(n_bytes, None),
        198: np.float64,  # double
        199: np.float32,  # float
        200: np.float32,  # float
    }

    dtype = type_to_dtype.get(type)
    if dtype is None:
        raise ValueError(f"Unsupported raster type: {type}")
    if isinstance(dtype, dict):
        if n_bytes not in dtype:
            raise ValueError(f"Invalid bit depth {depth} for type {type}")
        return dtype[n_bytes]
    return dtype


def post_process_data(data: np.ndarray, type: int, width: int) -> np.ndarray:
    """
    Apply post-processing for special raster types, particularly complex data.

    Args:
        data: Raw numpy array of image data
        type: Raster file type code
        width: Width of the image in pixels

    Returns:
        np.ndarray: Processed image data with correct complex representation
    """
    if type == 89:  # 4-bit complex signed
        real = np.right_shift(data, 4)
        imag = data & 0x0F
        real_sign = -0.125 * (real & 0x08) * (real & 0x07)
        imag_sign = -0.125 * (imag & 0x08) * (imag & 0x07)
        return real_sign + 1j * imag_sign

    elif type == 90:  # 4-bit complex unsigned
        real = np.right_shift(data, 4)
        imag = data & 0x0F
        return real + 1j * imag

    elif type == 93:  # complex signed bytes
        ind = np.arange(width)
        data = (-1 * (data & 0x80) // 0x80) * (data & 0x7F)
        return data[2 * ind] + 1j * data[2 * ind + 1]

    elif type == 94:  # complex unsigned bytes
        ind = 2 * np.arange(width)
        return data[ind] + 1j * data[ind + 1]

    elif type == 97 and data.dtype == np.int16:  # complex shorts
        ind = 2 * np.arange(width)
        return data[ind] + 1j * data[ind + 1]

    return data


def read_ras(filename: str) -> Tuple[np.ndarray, Optional[np.ndarray], RasterFooter]:
    """
    Read a Sun Raster file (.ras) and extract image data, colormap, and footer.

    File structure:
        - 32-byte header (8 long integers)
        - Optional colormap
        - Image data
        - Optional footer

    Args:
        filename: Path to the .ras file

    Returns:
        tuple: (
            np.ndarray: Image data,
            Optional[np.ndarray]: Colormap if present, else None,
            RasterFooter: Footer metadata if present, else None
        )
    """
    with open(filename, "rb") as f:
        # Read header - 8 long integers (32 bits each)
        header = struct.unpack(">8L", f.read(32))
        magic, width, height, depth, length, type, colormap_type, colormap_length = (
            header
        )

        # Read colormap if present
        cmap = None
        if colormap_type > 0 and colormap_length > 0:
            cmap = np.fromfile(f, dtype=np.uint8, count=colormap_length).reshape(-1, 3)

        # Calculate offset for image data
        n_bytes = depth // 8
        firstline = 0  # Default value from IDL code

        # Position file pointer
        data_offset = (
            8 * 4 + width * n_bytes * firstline + colormap_length
        )  # Same as IDL's calculation
        f.seek(data_offset)

        # Read binary data into numpy array
        dtype = get_dtype(type, depth)  # Get appropriate numpy dtype
        data = np.frombuffer(f.read(width * height * n_bytes), dtype=dtype).byteswap()

        # Reshape to match image dimensions
        image = data.reshape((height, width))

        # Reshape and post-process if needed
        if type in [89, 90, 93, 94, 97]:
            image = post_process_data(data, type, width)
        else:
            image = data.reshape((height, width))

        # Calculate footer size and position
        f.seek(0, 2)  # Seek to end
        file_size = f.tell()
        footer_size = file_size - (32 + length + colormap_length)  # 32 is header size

        # Read footer if present
        if footer_size > 0:
            f.seek(file_size - footer_size)
            footer_bytes = f.read(footer_size)
            footer = parse_footer(footer_bytes, type)
        else:
            footer = None

    return image, cmap, footer


def is_defined(footer_segment: str) -> bool:
    """
    Check if footer segment is valid (not null-terminated or empty).

    Args:
        footer_segment: String segment from footer

    Returns:
        bool: True if segment contains valid data
    """
    # Remove null bytes and whitespace
    cleaned = footer_segment.strip().replace("\x00", "")
    return bool(cleaned)


def safe_float(footer_segment: str, default: float = np.nan) -> float:
    """
    Safely convert footer segment to float with default value for invalid data.

    Args:
        footer_segment: String segment to convert
        default: Default value for invalid conversions (default: np.nan)

    Returns:
        float: Converted value or default if invalid
    """
    if not is_defined(footer_segment):
        return default
    try:
        return float(footer_segment.strip())
    except ValueError:
        return default


def parse_footer(footer_bytes: bytes, type: int) -> RasterFooter:
    """
    Parse binary footer data from Sun Raster file into structured RasterFooter object.

    The footer contains metadata about the raster image including geographic coordinates,
    pixel values statistics, and projection information. Footer format varies by type:
    - Type 100/200: DEM data with geographic metadata
    - Type 101: SAR data with specific metadata

    Format (16-byte blocks):
        0-15:   Map projection info
        16-17:  Pixel unit
        18-33:  Upper-left northing
        34-49:  Upper-left easting
        50-65:  Lower-right northing
        66-81:  Lower-right easting
        82-97:  Max value
        98-113: Min value
        114-129: Mean value
        130-145: Standard deviation
        146-161: Latitude posting
        162-177: Longitude posting
        178-193: Invalid value
        194-209: Number of invalid pixels
        210-225: Upper-left latitude
        226-241: Upper-left longitude
        ...etc

    Args:
        footer_bytes: Raw footer data as bytes
        type: Raster file type code (100, 101, 200 etc)

    Returns:
        RasterFooter: Parsed footer data in structured format

    Raises:
        ValueError: If footer data is invalid or cannot be parsed
        UnicodeDecodeError: If footer contains invalid ASCII characters

    Example:
        >>> footer_bytes = b'ESA GEO         SE      73.1155833...'
        >>> footer = parse_footer(footer_bytes, 200)
        >>> print(footer.ul_northing)
        73.1155833
    """
    footer_str = footer_bytes.decode("ascii")

    try:
        return RasterFooter(
            map=footer_str[0:16].strip() if is_defined(footer_str[0:16]) else "",
            pixel_unit=footer_str[16:18].strip()
            if is_defined(footer_str[16:18])
            else "",
            ul_northing=safe_float(footer_str[18:34]),
            ul_easting=safe_float(footer_str[34:50]),
            lr_northing=safe_float(footer_str[50:66]),
            lr_easting=safe_float(footer_str[66:82]),
            max_value=safe_float(footer_str[82:98]),
            min_value=safe_float(footer_str[98:114]),
            mean_value=safe_float(footer_str[114:130]),
            stdev=safe_float(footer_str[130:146]),
            lat_posting=safe_float(footer_str[146:162]),
            lon_posting=safe_float(footer_str[162:178]),
            inval=safe_float(footer_str[178:194]),
            inval_num=int(
                safe_float(footer_str[194:210], 0)
            ),  # Default to 0 for invalid count
            ul_lat=safe_float(footer_str[210:226]),
            ul_lon=safe_float(footer_str[226:242]),
            ur_lat=safe_float(footer_str[242:258]),
            ur_lon=safe_float(footer_str[258:274]),
            ll_lat=safe_float(footer_str[274:290]),
            ll_lon=safe_float(footer_str[290:306]),
            lr_lat=safe_float(footer_str[306:322]),
            lr_lon=safe_float(footer_str[322:338]),
            res_lat=safe_float(footer_str[338:354]),
            res_lon=safe_float(footer_str[354:370]),
        )
    except Exception as e:
        print(f"Error parsing footer: {str(e)}")
        raise


def write_geotiff(filename: str, image: np.ndarray, footer: RasterFooter) -> None:
    """
    Write image data to a GeoTIFF file with georeferencing information.

    Args:
        filename: Output GeoTIFF file path
        image: Image data as numpy array
        footer: Footer metadata for georeferencing
    """
    driver = gdal.GetDriverByName("GTiff")

    # Create output dataset
    out_ds = driver.Create(
        filename, image.shape[1], image.shape[0], 1, gdal.GDT_Float32
    )

    # Set geotransform
    pixel_width = (footer.lr_easting - footer.ul_easting) / image.shape[1]
    pixel_height = (footer.lr_northing - footer.ul_northing) / image.shape[0]
    geotransform = (
        footer.ul_easting,
        pixel_width,
        0,
        footer.ul_northing,
        0,
        pixel_height,
    )
    out_ds.SetGeoTransform(geotransform)

    # Set projection
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS("WGS84")
    out_ds.SetProjection(srs.ExportToWkt())

    # Write data
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(image)

    # Set nodata value
    out_band.SetNoDataValue(footer.inval)

    # Close dataset
    out_ds = None


def ras2geotiff(ras_filename: str, tiff_filename: str) -> None:
    """
    Convert a Sun Raster (.ras) file to GeoTIFF format.

    Args:
        ras_filename: Input .ras file path
        tiff_filename: Output GeoTIFF file path

    Example:
        >>> ras2geotiff('input.ras', 'output.tiff')
    """
    # Read the .ras file
    image, _, footer = read_ras(ras_filename)

    # Replace invalid values
    invalid_mask = image == footer.inval
    if invalid_mask.any():
        image = image.astype(np.float32)
        image[invalid_mask] = np.nan

    # Write as GeoTIFF
    write_geotiff(tiff_filename, image, footer)


# %%
# file = "/Users/imansour/Downloads/Islam Raw DEMs/TDM1_SAR__RAW_BIST_20170317T194223_1412139_3/RAW-DEM/tdm0_w037d30n73d12_54104_20170317T194223_003_DEM.ras"
# # Example usage
# image, cmap, footer = read_ras(file)


# plt.imshow(image)

# %%
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python script.py input.ras output.tiff")
        sys.exit(1)

    ras2geotiff(sys.argv[1], sys.argv[2])
