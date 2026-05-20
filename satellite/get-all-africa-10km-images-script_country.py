"""
Google Earth Engine Landsat-8 Image Export Pipeline
---------------------------------------------------
This script downloads Landsat-8 composite imagery for malaria
prediction locations across sub-Saharan Africa using Google Earth Engine.

Authentication:
Before running:
    earthengine authenticate
"""

# ============================================================
# Imports
# ============================================================

import time

import ee
import pandas as pd


# ============================================================
# Earth Engine Authentication
# ============================================================

# Modern authentication workflow
# Run once in terminal:
# earthengine authenticate

ee.Initialize()

print("Earth Engine initialized successfully.")


# ============================================================
# Configuration Parameters
# ============================================================

# Input CSV containing survey locations
# Use separate CSV files for separate countries to run scripts in parallel 
INPUT_CSV = "data/country_Guinea.csv"

# Google Drive export folder
DRIVE_FOLDER = "guinea"

# Temporal filtering
START_DATE = "2013-01-01"
END_DATE = "2022-12-31"

# Export dimensions
IMAGE_DIMS = "224x224"

# Half-width of image patch in degrees approximately corresponds to a 10 km × 10 km spatial window
PATCH_RESOLUTION = 0.04166665

# RGB bands from Landsat Collection 2
RGB_BANDS = ["SR_B4", "SR_B3", "SR_B2"]

# Cloud filtering
MAX_CLOUD_COVER = 20

# Export throttling
EXPORT_DELAY_SECONDS = 10
BATCH_PAUSE_INTERVAL = 100
BATCH_PAUSE_SECONDS = 600

# ============================================================
# Load Survey Locations
# ============================================================

print("Loading survey locations...")

locations = pd.read_csv(INPUT_CSV)

print(f"Loaded {len(locations)} locations.")


# ============================================================
# Landsat Composite Construction
# ============================================================

def build_landsat_composite():
    """
    Build cloud-reduced Landsat-8 composite.

    Returns
    -------
    ee.Image
        Composite image
    """

    print("Building Landsat composite...")

    # Old collection = ee.ImageCollection('LANDSAT/LC08/C01/T1')
    collection = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterDate(START_DATE, END_DATE)
        .filter(ee.Filter.lt("CLOUD_COVER", MAX_CLOUD_COVER))
    )

    # Median composite reduces cloud artefacts
    composite = (
        collection
        .median()
        .select(RGB_BANDS)
    )

    return composite


# ============================================================
# Geometry Construction
# ============================================================

def create_patch_geometry(lon, lat, resolution):
    """
    Create rectangular geometry around a location.

    Parameters
    ----------
    lon : float
        Longitude
    lat : float
        Latitude
    resolution : float
        Half-width of patch in degrees

    Returns
    -------
    ee.Geometry.Rectangle
    """

    return ee.Geometry.Rectangle([
        lon - resolution,
        lat - resolution,
        lon + resolution,
        lat + resolution
    ])


# ============================================================
# Filename Construction
# ============================================================

def create_filename(index, lon, lat, country_code):
    """
    Create reproducible filename for exports.
    """

    lon_str = str(lon).replace(".", "dd")
    lat_str = str(lat).replace(".", "dd")

    return (
        f"{index}_IMAGE_RGB_"
        f"{IMAGE_DIMS}_"
        f"{lon_str}_{lat_str}_{country_code}"
    )


# ============================================================
# Image Export Function
# ============================================================

def export_images_to_drive(
    dataframe,
    start_index=0,
    end_index=None
):
    """
    Export Landsat image patches to Google Drive.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Survey locations
    start_index : int
        Start index
    end_index : int
        End index
    """

    if end_index is None:
        end_index = len(dataframe)

    composite = build_landsat_composite()

    print(
        f"Starting exports from "
        f"{start_index} to {end_index}"
    )

    for i in range(start_index, end_index):

        # ----------------------------------------------------
        # Extract coordinates
        # ----------------------------------------------------

        lon = dataframe["lon"].iloc[i]
        lat = dataframe["lat"].iloc[i]

        country = str(
            dataframe["country_code"].iloc[i]
        )

        # ----------------------------------------------------
        # Create geometry
        # ----------------------------------------------------

        geometry = create_patch_geometry(
            lon,
            lat,
            PATCH_RESOLUTION
        )

        # ----------------------------------------------------
        # Construct filename
        # ----------------------------------------------------

        filename = create_filename(
            i,
            lon,
            lat,
            country
        )

        print(f"Submitting export: {filename}")

        # ----------------------------------------------------
        # Export image to Google Drive
        # ----------------------------------------------------

        task = ee.batch.Export.image.toDrive(
            image=composite,
            description=f"landsat_export_{i}",
            folder=DRIVE_FOLDER,
            fileNamePrefix=filename,
            region=geometry,
            dimensions=IMAGE_DIMS,
            maxPixels=1e13
        )

        task.start()

        # ----------------------------------------------------
        # Avoid Earth Engine rate limits
        # ----------------------------------------------------

        time.sleep(EXPORT_DELAY_SECONDS)

        if (i + 1) % BATCH_PAUSE_INTERVAL == 0:

            print(
                "Pausing temporarily to avoid "
                "Earth Engine task throttling..."
            )

            time.sleep(BATCH_PAUSE_SECONDS)


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":

    export_images_to_drive(
        dataframe=locations,
        start_index=0,
        end_index=len(locations)
    )

    print("All export tasks submitted.")
