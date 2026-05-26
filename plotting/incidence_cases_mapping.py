# ============================================================
# ADMIN-LEVEL MALARIA BURDEN AGGREGATION
#
# This script:
#
# 1. Loads malaria incidence rasters
# 2. Combines them with population rasters
# 3. Estimates malaria cases
# 4. Aggregates burden to Admin-0/Admin-1 levels
# 5. Generates summary tables and maps
#
# ============================================================

import os
import glob
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

# ============================================================
# DIRECTORIES
# ============================================================

INCIDENCE_DIR = "incidence_rasters"

POPULATION_DIR = "population_rasters"

ADMIN_DIR = "admin_rasters"

OUTPUT_DIR = "outputs"

TABLE_DIR = os.path.join(
    OUTPUT_DIR,
    "tables"
)

MAP_DIR = os.path.join(
    OUTPUT_DIR,
    "maps"
)

SUMMARY_DIR = os.path.join(
    OUTPUT_DIR,
    "summaries"
)

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(MAP_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

# ============================================================
# INPUT FILES
# ============================================================

ADMIN1_RASTER = os.path.join(
    ADMIN_DIR,
    "admin2023_1_MG_5K.tif"
)

ADMIN1_LOOKUP = os.path.join(
    ADMIN_DIR,
    "admin2023_1_MG_5K_config.csv"
)

POPULATION_RASTER = os.path.join(
    POPULATION_DIR,
    "ihme_corrected_worldpop_All_Ages_3_2023.tif"
)

# ============================================================
# LOAD ADMIN LOOKUP TABLE
# ============================================================

admin_lookup = pd.read_csv(
    ADMIN1_LOOKUP
)

print("Loaded admin lookup table")

# ============================================================
# LOAD ADMIN RASTER
# ============================================================

with rasterio.open(ADMIN1_RASTER) as src:
    admin_array = src.read(1)

# ============================================================
# LOAD POPULATION RASTER
# ============================================================

with rasterio.open(POPULATION_RASTER) as src:
    population_array = src.read(1)

# ============================================================
# GET INCIDENCE FILES
# ============================================================

incidence_files = sorted(
    glob.glob(
        os.path.join(
            INCIDENCE_DIR,
            "*.tif"
        )
    )
)

print(f"Found {len(incidence_files)} incidence rasters")

# ============================================================
# FUNCTION:
# COMPUTE ADMIN-LEVEL BURDEN
# ============================================================

def compute_admin_burden(
    incidence_array,
    population_array,
    admin_array,
    lookup_table
):
    """
    Estimate malaria burden by administrative unit.
    """

    # --------------------------------------------------------
    # COMPUTE ESTIMATED CASES
    # --------------------------------------------------------

    estimated_cases = (
        incidence_array *
        population_array
    )

    # --------------------------------------------------------
    # CREATE WORKING DATAFRAME
    # --------------------------------------------------------

    df = pd.DataFrame({
        "admin_id": admin_array.flatten(),
        "incidence": incidence_array.flatten(),
        "population": population_array.flatten(),
        "cases": estimated_cases.flatten()
    })

    # --------------------------------------------------------
    # CLEAN DATA
    # --------------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna()

    # --------------------------------------------------------
    # AGGREGATE TO ADMIN LEVEL
    # --------------------------------------------------------

    grouped = (
        df.groupby("admin_id")
        .agg({
            "cases": "sum",
            "population": "sum"
        })
        .reset_index()
    )

    # --------------------------------------------------------
    # POPULATION-WEIGHTED INCIDENCE
    # --------------------------------------------------------

    grouped["weighted_incidence"] = (
        grouped["cases"] /
        grouped["population"]
    )

    # --------------------------------------------------------
    # MERGE ADMIN NAMES
    # --------------------------------------------------------

    grouped = grouped.merge(
        lookup_table,
        left_on="admin_id",
        right_on="Value",
        how="left"
    )

    return grouped

# ============================================================
# PROCESS ALL REALISATIONS
# ============================================================

all_results = []

for idx, incidence_file in enumerate(incidence_files):

    print(
        f"Processing: {os.path.basename(incidence_file)}"
    )

    with rasterio.open(incidence_file) as src:
        incidence_array = src.read(1)

    # --------------------------------------------------------
    # COMPUTE ADMIN BURDEN
    # --------------------------------------------------------

    admin_results = compute_admin_burden(
        incidence_array=incidence_array,
        population_array=population_array,
        admin_array=admin_array,
        lookup_table=admin_lookup
    )

    admin_results["realisation"] = idx + 1

    all_results.append(admin_results)

# ============================================================
# COMBINE RESULTS
# ============================================================

final_df = pd.concat(
    all_results,
    ignore_index=True
)

# ============================================================
# SAVE ADMIN-LEVEL RESULTS
# ============================================================

output_csv = os.path.join(
    TABLE_DIR,
    "admin_level_estimated_cases.csv"
)

final_df.to_csv(
    output_csv,
    index=False
)

print(f"Saved results to: {output_csv}")

# ============================================================
# SUMMARY STATISTICS
# ============================================================

summary_df = (
    final_df
    .groupby(["NAME_0", "NAME_1"])
    .agg({
        "cases": ["mean", "std"],
        "weighted_incidence": ["mean", "std"]
    })
)

summary_output = os.path.join(
    SUMMARY_DIR,
    "admin_level_summary.csv"
)

summary_df.to_csv(summary_output)

print(
    f"Saved summary statistics to: {summary_output}"
)

# ============================================================
# CREATE MEAN INCIDENCE MAP
# ============================================================

print("Creating mean incidence map")

incidence_stack = []

for f in incidence_files:

    with rasterio.open(f) as src:

        incidence_stack.append(
            src.read(1)
        )

mean_incidence = np.nanmean(
    incidence_stack,
    axis=0
)

plt.figure(figsize=(12, 10))

plt.imshow(mean_incidence)

plt.title("Mean Malaria Incidence")

plt.colorbar(label="Incidence")

map_output = os.path.join(
    MAP_DIR,
    "mean_incidence_map.png"
)

plt.savefig(
    map_output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved map to: {map_output}")

