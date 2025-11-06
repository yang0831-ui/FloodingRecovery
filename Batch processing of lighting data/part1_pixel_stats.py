# -*- coding: utf-8 -*-
import os
import re
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support
import pyarrow as pa
import pyarrow.parquet as pq
import gc

import arcpy


# ------------------------- general utilities ------------------------- #
def sanitize_name(raw_name: str) -> str:
    # Simplify file name
    m = re.search(r"\.A(\d{7})\.", raw_name)
    date7 = m.group(1) if m else re.sub(r'[^0-9]', '', raw_name)[:7].ljust(7, '_')
    return f"d{date7}.dbf"


def _add_xy_id_field(fishnet_shp: str) -> None:
    # add xy_id = W/E + lon + N/S + lat to fishnet
    fields = [f.name for f in arcpy.ListFields(fishnet_shp)]
    if "xy_id" in fields:
        return

    arcpy.management.AddField(fishnet_shp, "xy_id", "TEXT", field_length=50)
    with arcpy.da.UpdateCursor(fishnet_shp, ["SHAPE@XY", "xy_id"]) as cur:
        for row in cur:
            x, y = row[0]
            xy_id = f"{'W' if x<0 else 'E'}{abs(x):.5f}{'S' if y<0 else 'N'}{abs(y):.5f}"
            row[1] = xy_id
            cur.updateRow(row)


def _flush_chunk(chunk: list[pd.DataFrame], accumulated: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate the current chunk DataFrame list → parse date → pivot → outer merge with accumulated wide table"""
    merged_long = pd.concat(chunk, ignore_index=True)

    # Parse date columns
    merged_long["year"]        = merged_long["filename"].str[1:5]
    merged_long["day_of_year"] = merged_long["filename"].str[5:8].astype(int)
    merged_long["date"]        = pd.to_datetime(
        merged_long["year"] + merged_long["day_of_year"].astype(str),
        format="%Y%j"
    )
    merged_long.drop(columns=["year", "day_of_year"], inplace=True)

    # Wide table
    wide_part = (
        merged_long
        .pivot(index="xy_id", columns="date", values="MEAN")
        .reset_index()
    )

    # First flush
    if accumulated is None:
        return wide_part

    # Subsequent flush: outer join by xy_id, append new date columns
    accumulated = accumulated.merge(wide_part, on="xy_id", how="outer", copy=False)
    return accumulated

def _cleanup_dbf(dbf_folder: str, dbf_files: list[str]) -> None:
    suffixes = (".dbf", ".cpg", ".dbf.xml")
    for dbf in dbf_files:
        stem = os.path.splitext(dbf)[0]
        for suf in suffixes:
            p = os.path.join(dbf_folder, stem + suf)
            if os.path.exists(p):
                try:
                    os.remove(p)
                    print(f"deleted: {p}")
                except Exception as e:
                    print(f"cannot delete {p} → {e}")
    print("✓ DBF cleanup complete")

def find_sample_raster(raster_folder_path):
        # Get all files in the folder and filter for common raster formats, such as .tif
    raster_files = [f for f in os.listdir(raster_folder_path) if f.endswith(('.tif', '.tiff'))]

    # Check if any raster files were found
    if not raster_files:
        raise FileNotFoundError("No .tif or .tiff raster files found in the specified folder.")

    # Select the first one as sample_raster
    sample_raster_path = os.path.join(raster_folder_path, raster_files[0])

    return sample_raster_path

# ----------------------- ① Generate Fishnet ------------------------ #
def build_shp_grid(
    sample_raster: str,
    out_shp: str,
    zone_field: str = "index",
    overwrite: bool = False
) -> str:

    out_shp = str(out_shp)
    if os.path.exists(out_shp) and not overwrite:
        print(f"fishnet already exists at: {out_shp}")
        _add_xy_id_field(out_shp)          # Ensure xy_id field exists
        return out_shp

    print(f"▶ Generating fishnet {out_shp}")
    ras = arcpy.Raster(sample_raster)
    arcpy.env.snapRaster  = ras
    arcpy.env.extent      = ras.extent
    arcpy.env.outputCoordinateSystem = ras.spatialReference

    cell_w, cell_h = ras.meanCellWidth, ras.meanCellHeight
    xmin, ymin, xmax, ymax = ras.extent.XMin, ras.extent.YMin, ras.extent.XMax, ras.extent.YMax

    arcpy.management.CreateFishnet(
        out_feature_class = out_shp,
        origin_coord      = f"{xmin} {ymin}",
        y_axis_coord      = f"{xmin} {ymin+1}",
        cell_width        = cell_w,
        cell_height       = cell_h,
        number_rows       = "0",
        number_columns    = "0",
        corner_coord      = f"{xmax} {ymax}",
        labels            = "NO_LABELS",
        template          = sample_raster,
        geometry_type     = "POLYGON"
    )

    arcpy.management.AddField(out_shp, zone_field, "LONG")
    arcpy.management.CalculateField(out_shp, zone_field, "!FID!", "PYTHON3")

    _add_xy_id_field(out_shp)
    print("✓ fishnet creation complete")
    return out_shp



# ---------------------- ② Zonal Statistics ----------------- #
def _zonal_one(
    tif_path: str,
    fishnet_shp: str,
    output_folder: str
) -> str:
    arcpy.CheckOutExtension("Spatial")
    fname = os.path.basename(tif_path)
    table_name = sanitize_name(fname.replace("_clip_filtered.tif", ""))
    out_table  = os.path.join(output_folder, table_name)

    if os.path.exists(out_table):
        return f"already exists: {table_name}"

    try:
        arcpy.sa.ZonalStatisticsAsTable(
            in_zone_data   = fishnet_shp,
            zone_field     = "xy_id",
            in_value_raster= tif_path,
            out_table      = out_table,
            ignore_nodata  = "DATA",
            statistics_type= "MEAN"
        )
        return f"finished: {table_name}"
    except Exception as e:
        return f"failed: {table_name} → {e}"


def run_zonal_statistics(
    raster_folder: str,
    fishnet_shp: str,
    output_folder: str,
    *,
    workers: int = 4        
) -> None:

    # --- Spatial extension ---
    if arcpy.CheckExtension("Spatial") != "Available":
        raise RuntimeError("Spatial Analyst extension is not available")
    arcpy.CheckOutExtension("Spatial")

    os.makedirs(output_folder, exist_ok=True)
    tif_list = [
        str(Path(raster_folder) / f)
        for f in os.listdir(raster_folder)
        if f.endswith("_clip_filtered.tif")
    ]

    if not tif_list:
        print("No *_clip_filtered.tif files found")
        return

    print(f"Starting ZonalStatistics for {len(tif_list)} rasters")

    # ---------- Parallel or Serial ----------
    if workers and workers > 1 and len(tif_list) > 1:
        with ProcessPoolExecutor(max_workers=workers) as exe:
            for res in exe.map(
                _zonal_one,
                tif_list,
                [fishnet_shp] * len(tif_list),
                [output_folder] * len(tif_list)
            ):
                print(res)
    else:
        for tif in tif_list:
            res = _zonal_one(tif, fishnet_shp, output_folder)
            print(res)

    print("All Zonal processing complete")


def merge_dbf_tables(
    dbf_folder: str,
    output_csv: str,
    *,
    chunk_size: int = 200,         
    delete_dbf: bool = False
) -> None:

    dbf_files = [f for f in os.listdir(dbf_folder) if f.lower().endswith(".dbf")]
    if not dbf_files:
        print("⚠ No .dbf files found. Skipping merge.")
        return

    wide_df: Optional[pd.DataFrame] = None     
    chunk: list[pd.DataFrame] = []            

    for idx, dbf in enumerate(dbf_files, 1):
        dbf_path = os.path.join(dbf_folder, dbf)
        try:
            arr = arcpy.da.TableToNumPyArray(dbf_path, "*")
            df  = pd.DataFrame(arr)
            df["filename"] = os.path.splitext(dbf)[0]
            chunk.append(df)
            print(f"Read {idx}/{len(dbf_files)} → {dbf}")
        except Exception as e:
            print(f"Skipped: {dbf} → {e}")

        # Flush when full
        if len(chunk) == chunk_size:
            wide_df = _flush_chunk(chunk, wide_df)    
            chunk.clear(); gc.collect()

    # Process the remaining records
    if chunk:
        wide_df = _flush_chunk(chunk, wide_df)
        chunk.clear(); gc.collect()

    if wide_df is None or wide_df.empty:
        print("⚠ The merging result is empty. Terminating.")
        return

    # ---------- Export to CSV ----------
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    wide_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✓ Exported CSV: {output_csv}")

    # ---------- Optional delete source files ----------
    if delete_dbf:
        _cleanup_dbf(dbf_folder, dbf_files)