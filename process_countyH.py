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



# --- 常量路径 ---
ROOT            = r"H:\FloodingRecovery\ProcessData"
RAW_VIIRS_ROOT  = r"H:\nighttimeLights"
COUNTY_CSV      = r"H:\FloodingRecovery\county_disaster_daily_matrix.csv"
COUNTY_SHP      = r"H:\FloodingRecovery\selectedCounty\selectedCounty.shp"
TILE_SHP        = r"H:\FloodingRecovery\BlackMarbleTiles\BlackMarbleTiles\BlackMarbleTiles.shp"
DATE_RANGE_CSV  = r"H:\FloodingRecovery\county_extraction_date_ranges.csv"


FISHNET_DIR     = os.path.join(ROOT, "fishnet")
CSV_DIR         = os.path.join(ROOT, "lighting_csv")
PROGRESS_FILE   = os.path.join(ROOT, "progress.txt")
LOG_FILE        = os.path.join(ROOT, "process.log")

os.makedirs(FISHNET_DIR, exist_ok=True)

# --- 外部模块函数 ---
from part1_clip_h52tiff import (
    viirs_hdf_to_clipped_tif,
    build_hdf_index,
)
from part1_pixel_stats import (
    build_shp_grid,
    find_sample_raster,
    run_zonal_statistics,
    merge_dbf_to_long_parquet,
    create_wide_csv_from_long_parquets,
)

# --- 日志 ---
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

def process_one_tile(tid: str, county_shp: str, sub_dir: Path, hdf_index: pd.DataFrame, date_ranges=None):
    """
    把原来 for tid in tile_list: 里的 3.1~3.3 全放进来
    注意不要再扫描父目录；直接用 sub_dir / "clip"
    """
    
    clip_dir = sub_dir / "clip"
    dbf_dir  = sub_dir / "dbf"
    clip_dir.mkdir(parents=True, exist_ok=True)
    dbf_dir.mkdir(exist_ok=True)

    # 1. 根据瓦片ID筛选
    tile_df = hdf_index[hdf_index["tile_id"] == tid].copy()
    
    # 2. 根据日期范围筛选
    if date_ranges:
        # 构建一个布尔掩码，标记所有落在任一日期范围内的行
        mask = pd.Series([False] * len(tile_df), index=tile_df.index)
        for start, end in date_ranges:
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            mask |= (tile_df['date'] >= start_dt) & (tile_df['date'] <= end_dt)
        
        tile_df = tile_df[mask]

    # 获取最终需要处理的文件路径列表
    files_to_process = tile_df["filepath"].tolist()
    if not files_to_process:
        logging.warning(f"[{tid}] 根据瓦片和日期筛选后，没有找到需要处理的文件。")
        return None
    
    

    # ——▶ 3.1： *单进程* 跑整目录
    viirs_hdf_to_clipped_tif(
        hdf_files_to_process = files_to_process,
        output_folder = str(clip_dir),
        cutline_shp   = county_shp,
        tile_filter   = tid,
        date_ranges   = date_ranges,
        dst_nodata    = -9999,
        overwrite     = False
    )

    # ——▶ 3.2 之后保持原来的串行逻辑
    sample_raster = find_sample_raster(str(clip_dir))
    if sample_raster is None:
        logging.warning(f"[{tid}] no raster – skipped")
        return None      # 返回值让县级函数判断是否成功

    fishnet_tile = sub_dir / f"fishnet_{tid}.shp"
    build_shp_grid(sample_raster, str(fishnet_tile))

    dbf_dir = sub_dir / "dbf"
    dbf_dir.mkdir(exist_ok=True)
    run_zonal_statistics(str(clip_dir), str(fishnet_tile), str(dbf_dir))

    # 定义输出的长格式parquet文件名
    long_parquet_tile = sub_dir / f"tile_{tid}_long.parquet"

    # 调用新的合并函数，取代 merge_dbf_tables
    merge_dbf_to_long_parquet(
        dbf_folder=str(dbf_dir),
        output_parquet=str(long_parquet_tile),
        delete_dbf=True # 推荐在中间步骤删除，节省空间
    )
    
    
    return fishnet_tile, long_parquet_tile



# --- 单县完整流程 ---
def export_county_polygon(code: str, out_shp: str) -> bool:


    original_code_for_log = code
    code5 = code.zfill(5)  # 确保 5 位

    # —— 清理潜在的 ArcPy 缓存 —— 
    try:
        arcpy.ClearWorkspaceCache_management()
        logging.info(f"[{original_code_for_log}] Cleared workspace cache at start.")
    except Exception as e_cache:
        logging.warning(f"[{original_code_for_log}] Failed to clear cache: {e_cache}")

    # —— 确认字段 CTFIPS以及空间参考 —— 
    try:
        fields = arcpy.ListFields(COUNTY_SHP, "CTFIPS")
        if not fields:
            logging.error(f"[{original_code_for_log}] Field 'CTFIPS' 不存在于 {COUNTY_SHP} 中。")
            return False
        desc = arcpy.Describe(COUNTY_SHP)
        spatial_ref = desc.spatialReference
    except Exception as e_desc:
        logging.error(f"[{original_code_for_log}] Describe/字段检测失败: {e_desc}")
        return False

    # —— 用 SearchCursor 把所有 CTFIPS = 'code5' 的几何读出来 —— 
    matching_geoms = []
    where_clause = f"CTFIPS = '{code5}'"

    try:
        with arcpy.da.SearchCursor(COUNTY_SHP, ["CTFIPS", "SHAPE@"], where_clause=where_clause) as cursor:
            for row in cursor:
                # 这里再做一次“保险”：即便 DBF 里存的是 '06079'、'6079' 或其它格式，
                # 也按字符串 zfill 比对一遍，确保只能拿 code5 对应那条“本州/本县”。
                if str(row[0]).zfill(5) == code5:
                    matching_geoms.append(row[1])
    except Exception as e_sc:
        logging.error(f"[{original_code_for_log}] 用 SearchCursor 检索失败: {e_sc}")
        return False

    if not matching_geoms:
        logging.warning(f"[{original_code_for_log}] 在 {COUNTY_SHP} 中找不到 CTFIPS = '{code5}' 的要素。")
        return False

    # —— 如果同名的 out_shp 已存在 —— 
    if arcpy.Exists(out_shp):
        try:
            arcpy.Delete_management(out_shp)
        except Exception as e_del:
            logging.warning(f"[{original_code_for_log}] 删除已存在的 {out_shp} 失败: {e_del}")

    # —— CreateFeatureclass_management 新建一个空要素类 —— 
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
        logging.error(f"[{original_code_for_log}] CreateFeatureclass_management 失败: {e_cc}")
        return False

    # —— InsertCursor 把 matching_geoms的几何插入到 out_shp —— 
    try:
        with arcpy.da.InsertCursor(out_shp, ["SHAPE@"]) as ic:
            for geom in matching_geoms:
                ic.insertRow([geom])
    except Exception as e_ic:
        logging.error(f"[{original_code_for_log}] InsertCursor 写要素失败: {e_ic}")
        # 如果写入出错，也删掉 out_shp，不留空图层
        if arcpy.Exists(out_shp):
            try:
                arcpy.Delete_management(out_shp)
            except:
                pass
        return False

    logging.info(f"[{original_code_for_log}] 成功生成 {out_shp}，共 {len(matching_geoms)} 个要素。")

    # —— 再次清理缓存 —— 
    try:
        arcpy.ClearWorkspaceCache_management()
    except:
        pass

    return True



def find_tile_ids(county_shp: str) -> list[str]:

    import arcpy
    from pathlib import Path

    original = Path(county_shp).stem

    # —— 验证 county_shp 是否存在 —— 
    try:
        desc_county = arcpy.Describe(county_shp)
        sr_county   = desc_county.spatialReference
        ext_c       = desc_county.extent
        logging.info(f"[{original}] County describe: SR={sr_county.name}, Extent=({ext_c.XMin:.6f}, {ext_c.YMin:.6f}, {ext_c.XMax:.6f}, {ext_c.YMax:.6f})")
    except Exception as e_desc:
        logging.error(f"[{original}] Describe({county_shp}) 失败: {e_desc}")
        return []

    # —— 读取 county_shp 内所有几何 —— 
    county_geoms = []
    try:
        with arcpy.da.SearchCursor(county_shp, ["SHAPE@"]) as cursor:
            for row in cursor:
                geom = row[0]
                if geom is not None:
                    county_geoms.append(geom)
    except Exception as e_sc:
        logging.error(f"[{original}] 在 {county_shp} 中读取几何失败: {e_sc}")
        return []

    if not county_geoms:
        logging.warning(f"[{original}] {county_shp} 中没有任何几何要素。")
        return []

    # —— 检查几何交集 —— 
    tile_ids = set()
    try:
        with arcpy.da.SearchCursor(TILE_SHP, ["TileID", "SHAPE@"]) as t_cursor:
            for t_row in t_cursor:
                tile_id   = t_row[0]
                tile_geom = t_row[1]
                if tile_geom is None:
                    continue
                # 对每个 county_geom 做 intersect
                for county_geom in county_geoms:
                    try:
                        inter = county_geom.intersect(tile_geom, 4)  # 4 = esriGeometryPolygon
                        if inter is not None and inter.area > 0:
                            tile_ids.add(tile_id)
                            break  
                    except Exception as e_int:
                        logging.warning(f"[{original}] County-Geom.intersect(TileID={tile_id}) 异常: {e_int}")
                        continue
    except Exception as e_tcur:
        logging.error(f"[{original}] 遍历瓦片 {TILE_SHP} 失败: {e_tcur}")
        return []

    if not tile_ids:
        logging.warning(f"[{original}] 经过几何运算，没有找到任何相交瓦片 (No tile found).")
    else:
        sorted_ids = sorted(tile_ids)
        logging.info(f"[{original}] 通过几何运算找到瓦片: {sorted_ids}")
        return sorted_ids

    return []


def merge_tile_csvs_incremental(tile_csvs, csv_final, chunksize=5000):

    os.makedirs(os.path.dirname(csv_final), exist_ok=True)
    header_written = False

    if len(tile_csvs) == 1:
        os.makedirs(os.path.dirname(csv_final), exist_ok=True)
        shutil.copy(tile_csvs[0], csv_final)
        print(f"✓ 只有一个文件，已复制：{tile_csvs[0]} → {csv_final}")
        return

    with open(csv_final, 'w', encoding='utf-8-sig', newline='') as fout:
        for csvp in tile_csvs:
            print(f"合并中：{csvp}")
            # 按块读取，避免一次性全部载入
            for chunk in pd.read_csv(csvp, dtype={"xy_id": str}, chunksize=chunksize):
                chunk.to_csv(
                    fout,
                    index=False,
                    header=not header_written,
                    encoding='utf-8-sig'
                )
                header_written = True
    print(f"✓ 已写入最终 CSV：{csv_final}")



from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging, shutil, arcpy     # 其余 import 与原来相同

def process_one_county(code: str, HDF_INDEX_DF) -> None:
    tmp_dir       = Path(ROOT, code)
    fishnet_final = Path(FISHNET_DIR, f"{code}.shp")
    csv_final     = Path(CSV_DIR,     f"{code}.csv")

    success = False                 # 用小写，更符合 PEP8
    
    try:
        tmp_dir.mkdir(exist_ok=True)

        # ---------- 0. 县界 ----------
        county_shp = tmp_dir / f"{code}.shp"
        if not export_county_polygon(code, str(county_shp)):
            return

        # ---------- 1. 找 tile ----------
        tile_ids = find_tile_ids(str(county_shp))
        if not tile_ids:
            logging.warning(f"[{code}] No tile found – skipped")
            return
        logging.info(f"[{code}] Tiles: {tile_ids}")

         # <-- 变化点：变量名更新，以反映内容是Parquet文件 -->
        tile_fishnets, tile_parquets = [], []     

        # ---------- 2. 并行裁剪 ----------
        date_ranges = DATE_RANGES.get(code.zfill(5))
        with ProcessPoolExecutor(max_workers=4) as ex:

            fut2tid = {
                ex.submit(
                    process_one_tile, tid, str(county_shp),
                    tmp_dir / f"tile_{tid}",    
                    HDF_INDEX_DF,             
                    date_ranges                             
                ): tid
                for tid in tile_ids
            }

            for fut in as_completed(fut2tid):
                tid = fut2tid[fut]
                try:
                    res = fut.result()        # res 期望返回 (fishnet, csv) 或 None
                    if not res:
                        logging.warning(f"[{code}] Tile {tid} skipped by worker")
                        continue

                    fishnet_tile, long_parquet_tile = map(Path, res)
                    if fishnet_tile.exists():
                        tile_fishnets.append(str(fishnet_tile))
                    else:
                        logging.warning(f"[{code}] {fishnet_tile} missing – ignored")

                    # 确认 long_parquet_tile 存在
                    if long_parquet_tile.exists():
                        tile_parquets.append(str(long_parquet_tile))
                    else:
                        logging.warning(f"[{code}] {long_parquet_tile} missing – ignored")

                except Exception as e:
                    logging.exception(f"[{code}] Tile {tid} failed: {e}")



        # ---------- 4. 合并 ----------
        if not tile_fishnets:
            logging.warning(f"[{code}] No valid fishnets – skipped")
            return                      # 没有任何 shp，就没有继续的必要

        # 4.1 Fishnet 合并
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

        # 4.2 从所有长格式Parquet合并为最终的宽格式CSV
        # 4.2 从长格式Parquet合并为最终的宽格式CSV
        if not tile_parquets:
            logging.warning(f"[{code}] No Parquet files produced – skipped CSV creation.")
            return
        try:
            create_wide_csv_from_long_parquets(tile_parquets, str(csv_final))
        except Exception as e:
            logging.exception(f"[{code}] Failed to create wide CSV from Parquets: {e}")
            return

        logging.info(f"[{code}] Completed successfully with {len(tile_parquets)} tiles processed.")
        success = True
        mark_done(code)

    except Exception as e:
        logging.exception(f"[{code}] FAILED: {e}")

    finally:
        # 清理或保留临时目录
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
    HDF_INDEX_DF = build_hdf_index(RAW_VIIRS_ROOT)


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
            logging.warning(f"[{code}] 不在 {COUNTY_SHP} 中，跳过。")

    # 剔除已经做过的 
    done_set = load_done_set()
    pending  = [c for c in to_keep if c not in done_set]

    logging.info(f"总共有 {len(all_from_csv)} 个县；shapefile 可匹配 {len(to_keep)}；已完成 {len(done_set)}，剩余 {len(pending)}")

    for code in pending:
        logging.info(f"[{code}] === Start ===")
        process_one_county(code,HDF_INDEX_DF)

if __name__ == "__main__":
    main()