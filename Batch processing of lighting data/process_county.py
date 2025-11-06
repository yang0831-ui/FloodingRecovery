# -*- coding: utf-8 -*-
import os
import shutil
import logging
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob, re, shutil
import numpy as np
import pyarrow.dataset as ds

import arcpy
import pandas as pd


state_name = 'nj'

# --- Constant Path ---
ROOT            = rf"E:\ProcessData\ProcessData_{state_name.upper()}"
RAW_VIIRS_ROOT  = r"F:\数据\Flood_lighting\NY"
COUNTY_CSV      = rf"E:\OneDrive - National University of Singapore\研二下\Flooding\{state_name.lower()}_counties.csv"
COUNTY_SHP      = r"E:\National University of Singapore\Yang Yang - flooding\Raw Data\SHP\selectedCounty.shp"
TILE_SHP        = r"E:\National University of Singapore\Yang Yang - flooding\Raw Data\SHP\BlackMarbleTiles.shp"
DATE_RANGE_CSV = r"E:\OneDrive - National University of Singapore\研二下\Flooding\county_extraction_date_ranges.csv"


FISHNET_DIR     = os.path.join(ROOT, "fishnet")
CSV_DIR         = os.path.join(ROOT, "lighting_csv")
PROGRESS_FILE   = os.path.join(ROOT, "progress.txt")
LOG_FILE        = os.path.join(ROOT, "process.log")

os.makedirs(FISHNET_DIR, exist_ok=True)

# --- External module functions ---
from part1_clip_h52tiff import viirs_hdf_to_clipped_tif
from part1_pixel_stats import (
    build_shp_grid,
    find_sample_raster,
    run_zonal_statistics,
    merge_dbf_tables
)

# --- logging ---
os.makedirs(FISHNET_DIR, exist_ok=True)
os.makedirs(CSV_DIR,     exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8"
)

def load_done_set() -> set[str]:
    if not os.path.exists(PROGRESS_FILE):
        return set()
    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f)

def mark_done(code: str) -> None:
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(f"{code}\n")




_df_rng = (
    pd.read_csv(DATE_RANGE_CSV, dtype=str, parse_dates=["startDate", "endDate"])
      .assign(countyCode=lambda d: d["countyCode"].str.zfill(5))
)
DATE_RANGES = (
    _df_rng.groupby("countyCode")
           .apply(lambda g: list(zip(g["startDate"].dt.date, g["endDate"].dt.date)))
           .to_dict()
)

def process_one_tile(tid: str, county_shp: str, sub_dir: Path,date_ranges=None):

    clip_dir = sub_dir / "clip"
    dbf_dir  = sub_dir / "dbf"
    clip_dir.mkdir(parents=True, exist_ok=True)
    dbf_dir.mkdir(exist_ok=True)


    viirs_hdf_to_clipped_tif(
        input_folder  = RAW_VIIRS_ROOT, 
        output_folder = str(clip_dir),
        cutline_shp   = county_shp,
        tile_filter   = tid,
        date_ranges   = date_ranges,
        dst_nodata    = -9999,
        overwrite     = False
    )

    sample_raster = find_sample_raster(str(clip_dir))
    if sample_raster is None:
        logging.warning(f"[{tid}] no raster – skipped")
        return None     

    fishnet_tile = sub_dir / f"fishnet_{tid}.shp"
    build_shp_grid(sample_raster, str(fishnet_tile))

    dbf_dir = sub_dir / "dbf"
    dbf_dir.mkdir(exist_ok=True)
    run_zonal_statistics(str(clip_dir), str(fishnet_tile), str(dbf_dir))

    csv_tile = sub_dir / f"csv_{tid}.csv"
    merge_dbf_tables(str(dbf_dir), str(csv_tile), delete_dbf=False)

    return fishnet_tile, csv_tile



# --- single county process ---
def export_county_polygon(code: str, out_shp: str) -> bool:


    original_code_for_log = code
    code5 = code.zfill(5) 

    # Clearing Potential ArcPy Cache
    try:
        arcpy.ClearWorkspaceCache_management()
        logging.info(f"[{original_code_for_log}] Cleared workspace cache at start.")
    except Exception as e_cache:
        logging.warning(f"[{original_code_for_log}] Failed to clear cache: {e_cache}")

    # Confirm the fields CTFIPS and spatial reference
    try:
        fields = arcpy.ListFields(COUNTY_SHP, "CTFIPS")
        if not fields:
            logging.error(f"[{original_code_for_log}] Field 'CTFIPS' does not exist in {COUNTY_SHP}.")
            return False
        desc = arcpy.Describe(COUNTY_SHP)
        spatial_ref = desc.spatialReference
    except Exception as e_desc:
        logging.error(f"[{original_code_for_log}] Describe/Field check failed: {e_desc}")
        return False

    # —— Use SearchCursor to read all geometries where CTFIPS = 'code5' —— 
    matching_geoms = []
    where_clause = f"CTFIPS = '{code5}'"

    try:
        with arcpy.da.SearchCursor(COUNTY_SHP, ["CTFIPS", "SHAPE@"], where_clause=where_clause) as cursor:
            for row in cursor:
                if str(row[0]).zfill(5) == code5:
                    matching_geoms.append(row[1])
    except Exception as e_sc:
        logging.error(f"[{original_code_for_log}] Failed to retrieve using SearchCursor: {e_sc}")
        return False

    if not matching_geoms:
        logging.warning(f"[{original_code_for_log}] No features found in {COUNTY_SHP} where CTFIPS = '{code5}'.")
        return False
    # —— Delete existing out_shp if it exists ——
    if arcpy.Exists(out_shp):
        try:
            arcpy.Delete_management(out_shp)
        except Exception as e_del:
            logging.warning(f"[{original_code_for_log}] Failed to delete existing {out_shp}: {e_del}")

    # —— CreateFeatureclass_management —— 
    out_dir  = os.path.dirname(out_shp)
    out_name = os.path.basename(out_shp)
    try:
        arcpy.CreateFeatureclass_management(
            out_path=out_dir,
            out_name=out_name,
            geometry_type="POLYGON",
            spatial_reference=spatial_ref
        )
    except Exception as e_cc:
        logging.error(f"[{original_code_for_log}] CreateFeatureclass_management failed: {e_cc}")
        return False

    # —— InsertCursor inserts the geometries of matching_geoms into out_shp —— 
    try:
        with arcpy.da.InsertCursor(out_shp, ["SHAPE@"]) as ic:
            for geom in matching_geoms:
                ic.insertRow([geom])
    except Exception as e_ic:
        logging.error(f"[{original_code_for_log}] InsertCursor failed: {e_ic}")

        if arcpy.Exists(out_shp):
            try:
                arcpy.Delete_management(out_shp)
            except:
                pass
        return False

    logging.info(f"[{original_code_for_log}] Successfully generated {out_shp}, with {len(matching_geoms)} features.")

    # —— Clear cache again —— 
    try:
        arcpy.ClearWorkspaceCache_management()
    except:
        pass

    return True



def find_tile_ids(county_shp: str) -> list[str]:

    import arcpy
    from pathlib import Path

    original = Path(county_shp).stem

    # —— Validate county_shp exists —— 
    try:
        desc_county = arcpy.Describe(county_shp)
        sr_county   = desc_county.spatialReference
        ext_c       = desc_county.extent
        logging.info(f"[{original}] County describe: SR={sr_county.name}, Extent=({ext_c.XMin:.6f}, {ext_c.YMin:.6f}, {ext_c.XMax:.6f}, {ext_c.YMax:.6f})")
    except Exception as e_desc:
        logging.error(f"[{original}] Describe({county_shp}) failed: {e_desc}")
        return []

    # —— Read all geometries within county_shp —— 
    county_geoms = []
    try:
        with arcpy.da.SearchCursor(county_shp, ["SHAPE@"]) as cursor:
            for row in cursor:
                geom = row[0]
                if geom is not None:
                    county_geoms.append(geom)
    except Exception as e_sc:
        logging.error(f"[{original}] Failed to read geometries from {county_shp}: {e_sc}")
        return []

    if not county_geoms:
        logging.warning(f"[{original}] No geometries found in {county_shp}.")
        return []

    # —— Check for geometric intersection —— 
    tile_ids = set()
    try:
        with arcpy.da.SearchCursor(TILE_SHP, ["TileID", "SHAPE@"]) as t_cursor:
            for t_row in t_cursor:
                tile_id   = t_row[0]
                tile_geom = t_row[1]
                if tile_geom is None:
                    continue
                # Check the geometric intersection and perform the intersect operation for each county_geom
                for county_geom in county_geoms:
                    try:
                        inter = county_geom.intersect(tile_geom, 4)  # 4 = esriGeometryPolygon
                        if inter is not None and inter.area > 0:
                            tile_ids.add(tile_id)
                            break  
                    except Exception as e_int:
                        logging.warning(f"[{original}] County-Geom.intersect(TileID={tile_id}) failed: {e_int}")
                        continue
    except Exception as e_tcur:
        logging.error(f"[{original}] Failed to iterate tiles {TILE_SHP}: {e_tcur}")
        return []

    if not tile_ids:
        logging.warning(f"[{original}] No tile found after geometric operations.")
    else:
        sorted_ids = sorted(tile_ids)
        logging.info(f"[{original}] Found tiles through geometric operations: {sorted_ids}")
        return sorted_ids

    return []


def merge_tile_csvs_incremental(tile_csvs, csv_final, chunksize=5000):
    os.makedirs(os.path.dirname(csv_final) or ".", exist_ok=True)

    # Filter to retain only the existing files
    tile_csvs = [p for p in tile_csvs if os.path.exists(p)]
    if not tile_csvs:
        raise ValueError("merge_tile_csvs_incremental: No available input CSV.")

    # Only one file: copy directly (keep behavior consistent with old version)
    if len(tile_csvs) == 1:
        shutil.copy(tile_csvs[0], csv_final)
        print(f"✓ Only one file, copied: {tile_csvs[0]} → {csv_final}")
        return

    # —— 1) Scan all headers and build the "union columns all_cols" —— #
    def _read_header(path):
        return list(pd.read_csv(path, nrows=0).columns)

    try:
        all_cols = _read_header(tile_csvs[0])
    except Exception as e:
        raise RuntimeError(f"Failed to read header of first CSV: {tile_csvs[0]}, {e}")

    if not all_cols:
        raise RuntimeError(f"First CSV has no columns: {tile_csvs[0]}")

    seen = set(all_cols)
    for p in tile_csvs[1:]:
        cols = _read_header(p)
        for c in cols:
            if c not in seen:
                all_cols.append(c)
                seen.add(c)

    #  2) Determine the final column order: xy_id comes first #
    xy_col = "xy_id" if "xy_id" in all_cols else None

    # Try to parse non-xy_id columns as datetime
    others = [c for c in all_cols if c != xy_col]
    parsed_dt = pd.to_datetime(others, errors="coerce", format=None)  # Allow multiple formats, e.g., 2019/1/6 0:00
    date_cols = [col for col, dt in zip(others, parsed_dt) if not pd.isna(dt)]
    non_date_cols = sorted([col for col, dt in zip(others, parsed_dt) if pd.isna(dt)])

    # Sort the parsable date columns by the parsed time
    date_cols_sorted = [col for _, col in sorted(zip(parsed_dt[[others.index(c) for c in date_cols]], date_cols))]

    final_cols = ([xy_col] if xy_col else []) + date_cols_sorted + non_date_cols

    # —— 3) Stream merge according to final_cols —— #
    header_written = False
    with open(csv_final, 'w', encoding='utf-8-sig', newline='') as fout:
        for idx, csvp in enumerate(tile_csvs):
            print(f"[{idx+1}/{len(tile_csvs)}] Merging (aligning and sorting columns): {csvp}")
            for chunk in pd.read_csv(csvp, chunksize=chunksize):
                if xy_col and xy_col in chunk.columns:
                    chunk[xy_col] = chunk[xy_col].astype(str)

                # Fill missing columns with NA
                for col in final_cols:
                    if col not in chunk.columns:
                        chunk[col] = pd.NA

                # Reorder columns
                chunk = chunk[final_cols]

                # Write out (only write header the first time)
                chunk.to_csv(
                    fout,
                    index=False,
                    header=(not header_written),
                    encoding='utf-8-sig'
                )
                header_written = True

    print(f"✓ Final CSV written: {csv_final}")



from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging, shutil, arcpy    

def process_one_county(code: str) -> None:
    tmp_dir       = Path(ROOT, code)
    fishnet_final = Path(FISHNET_DIR, f"{code}.shp")
    csv_final     = Path(CSV_DIR,     f"{code}.csv")

    success = False               

    try:
        tmp_dir.mkdir(exist_ok=True)

        # ---------- 0. county border ----------
        county_shp = tmp_dir / f"{code}.shp"
        if not export_county_polygon(code, str(county_shp)):
            return

        # ---------- 1. Find tile ----------
        tile_ids = find_tile_ids(str(county_shp))
        if not tile_ids:
            logging.warning(f"[{code}] No tile found – skipped")
            return
        logging.info(f"[{code}] Tiles: {tile_ids}")

        tile_fishnets, tile_csvs = [], []     

        # ---------- 2. Parallel clipping ----------
        date_ranges = DATE_RANGES.get(code.zfill(5))
        with ProcessPoolExecutor(max_workers=4) as ex:

            fut2tid = {
                ex.submit(
                    process_one_tile, tid, str(county_shp),
                    tmp_dir / f"tile_{tid}",                
                    date_ranges                             
                ): tid
                for tid in tile_ids
            }

            for fut in as_completed(fut2tid):
                tid = fut2tid[fut]
                try:
                    res = fut.result()      
                    if not res:
                        logging.warning(f"[{code}] Tile {tid} skipped by worker")
                        continue

                    fishnet_tile, csv_tile = map(Path, res)
                    if fishnet_tile.exists():
                        tile_fishnets.append(str(fishnet_tile))
                    else:
                        logging.warning(f"[{code}] {fishnet_tile} missing – ignored")

                    if csv_tile.exists():
                        tile_csvs.append(str(csv_tile))
                    else:
                        logging.warning(f"[{code}] {csv_tile} missing – ignored")

                except Exception as e:
                    logging.exception(f"[{code}] Tile {tid} failed: {e}")



        # ---------- 4. Merge ----------
        if not tile_fishnets:
            logging.warning(f"[{code}] No valid fishnets – skipped")
            return                      

        # 4.1 Fishnet merge
        try:
            if fishnet_final.exists():
                arcpy.Delete_management(str(fishnet_final))
            if len(tile_fishnets) == 1:
                arcpy.management.CopyFeatures(tile_fishnets[0], str(fishnet_final))
            else:
                arcpy.management.Merge(tile_fishnets, str(fishnet_final))
        except Exception as e:
            logging.exception(f"[{code}] Merge fishnet failed: {e}")
            return

        # 4.2 CSV merge – **Again, only take truly existing files**
        valid_csvs = [c for c in tile_csvs if Path(c).exists()]
        if not valid_csvs:
            logging.warning(f"[{code}] No CSV produced – skipped")
            return
        try:
            merge_tile_csvs_incremental(valid_csvs, csv_final)
        except Exception as e:
            logging.exception(f"[{code}] Merge CSV failed: {e}")
            return

        logging.info(f"[{code}] Completed successfully with {len(valid_csvs)} tiles")
        success = True
        mark_done(code)

    except Exception as e:
        logging.exception(f"[{code}] FAILED: {e}")

    finally:
        try:
            arcpy.ClearWorkspaceCache_management()
        except Exception:
            pass

        if success:
            try:
                shutil.rmtree(tmp_dir)
                logging.info(f"[{code}] Temp folder deleted.")
            except Exception as cleanup_err:
                logging.warning(f"[{code}] !! Failed to delete temp folder: {cleanup_err}")
        else:
            logging.warning(f"[{code}] Kept temp folder for debug.")

            

def main() -> None:
    mp.freeze_support()


    all_ctfips = set()
    with arcpy.da.SearchCursor(COUNTY_SHP, ["CTFIPS"]) as cursor:
        for row in cursor:
            all_ctfips.add(row[0])

    df = pd.read_csv(COUNTY_CSV, dtype=str)
    df["countyCode"] = df["countyCode"].str.strip().str.zfill(5)
    all_from_csv = set(df["countyCode"]) 
    to_keep = sorted(all_from_csv & all_ctfips)
    to_skip = sorted(all_from_csv - all_ctfips)

    if to_skip:
        for code in to_skip:
            logging.warning(f"[{code}] Not in {COUNTY_SHP}, skipping.")
 
    done_set = load_done_set()
    pending  = [c for c in to_keep if c not in done_set]

    logging.info(f"Total counties: {len(all_from_csv)}; Shapefile matches: {len(to_keep)}; Completed: {len(done_set)}; Remaining: {len(pending)}")

    for code in pending:
        logging.info(f"[{code}] === Start ===")
        process_one_county(code)

if __name__ == "__main__":
    main()