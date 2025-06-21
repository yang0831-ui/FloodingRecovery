import re
from pathlib import Path
from typing import List, Optional
from osgeo import gdal, ogr
import numpy as np
import datetime as _dt
import pandas as pd
# ------------------------------------------------------------------
gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
gdal.SetConfigOption("GDAL_CACHEMAX", "4096")   
gdal.PushErrorHandler("CPLQuietErrorHandler")

def build_hdf_index(root_folder: str) -> pd.DataFrame:
    """
    扫描指定的根目录，为所有HDF文件创建一个元数据索引。
    这个函数应该在整个程序开始时只运行一次。

    Args:
        root_folder: 包含VIIRS HDF文件的根目录。

    Returns:
        一个包含文件信息的 pandas DataFrame，列为 ['filepath', 'tile_id', 'date']。
    """
    print("正在构建HDF文件索引，请稍候...")
    file_records = []
    
    # 正则表达式用于从文件名中提取信息
    # VNP46A1.A2022001.h08v05.001.2022002093516.h5
    date_pattern = re.compile(r"\.A(\d{4})(\d{3})\.")
    tile_pattern = re.compile(r"(h\d{2}v\d{2})")

    for h5_path in Path(root_folder).rglob("*.h*"):
        filename = h5_path.name
        
        # 提取日期
        date_match = date_pattern.search(filename)
        if not date_match:
            continue # 文件名不合规，跳过
        
        file_date = _dt.datetime.strptime(date_match.group(1) + date_match.group(2), "%Y%j").date()

        # 提取瓦片ID
        tile_match = tile_pattern.search(filename)
        tile_id = tile_match.group(1) if tile_match else None
        
        file_records.append({
            "filepath": str(h5_path),
            "tile_id": tile_id,
            "date": file_date
        })

    if not file_records:
        raise FileNotFoundError(f"在目录 {root_folder} 中没有找到任何HDF文件。")

    df = pd.DataFrame(file_records)
    df['date'] = pd.to_datetime(df['date']) # 转换为pandas的datetime对象以便于比较
    
    print(f"索引构建完成，共找到 {len(df)} 个有效文件。")
    return df

# ------------------------------------------------------------------
#  util: 找到子数据集路径
# ------------------------------------------------------------------
def _find_subdataset(subdatasets, keywords) -> Optional[str]:
    kw_lower = [k.lower() for k in keywords]
    for path, desc in subdatasets:
        line = (path + desc).lower()
        if any(k in line for k in kw_lower):
            return path
    return None

gdal.UseExceptions()

_tile_re = re.compile(r"h(\d{2})v(\d{2})", re.I)


def _get_tile_hv(h5_path: Path, meta: dict) -> tuple[int, int]:
    # 优先从文件名解析 h/v，回退到元数据
    m = _tile_re.search(h5_path.name)
    if m:
        return int(m.group(1)), int(m.group(2))

    try:
        return int(meta["HorizontalTileNumber"]), int(meta["VerticalTileNumber"])
    except KeyError:
        raise RuntimeError("无法获取 tile 号（文件名和元数据里都找不到 h/v）")


def viirs_hdf_to_clipped_tif(
    hdf_files_to_process: List[str],
    output_folder: str,
    cutline_shp: str,
    *,
    tile_filter: Optional[str] = None,
    date_ranges: Optional[list[tuple]] = None,
    dst_nodata: float = -9999.0,
    overwrite: bool = False,
) -> List[str]:
    
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ogr.Open(cutline_shp):
        raise RuntimeError(f"无法打开 cutline：{cutline_shp}")

    tif_written: List[str] = []
    for h5_file in sorted(hdf_files_to_process):
        h5_path = Path(h5_file)

        out_tif = out_dir / f"{h5_path.stem}_clip_filtered.tif"
        if out_tif.exists():
            if overwrite:                          # 要强制覆盖
                gdal.Unlink(str(out_tif))          
                print(f"• 发现旧文件，已删除：{out_tif.name}")
            else:                                  # 不覆盖，跳过
                print(f"• 跳过已存在：{out_tif.name}")
                tif_written.append(str(out_tif))
                continue

        # print(f"▶ 处理 {h5_path.name}")

        # ---------- 提前占位，避免 finally 报 UnboundLocalError ----------
        vrt_light = vrt_qual = vrt_cloud = vrt_snow = None

        try:
            h5_ds = gdal.Open(str(h5_path), gdal.GA_ReadOnly)
            subdatasets = h5_ds.GetSubDatasets()
            if not subdatasets:
                print("  × 无子数据集，跳过")
                continue

            light_path = _find_subdataset(subdatasets, ["DNB_BRDF-Corrected_NTL"])
            qual_path  = _find_subdataset(subdatasets, ["Mandatory_Quality_Flag"])
            cloud_path = _find_subdataset(subdatasets, ["QF_Cloud_Mask"])
            snow_path  = _find_subdataset(subdatasets, ["Snow_Flag"])
            if None in (light_path, qual_path, cloud_path, snow_path):
                print("  × 缺少必要子波段，跳过")
                continue

            light_ds = gdal.Open(light_path)

            # ---------- 取 h/v → 计算外包范围 ----------
            meta = h5_ds.GetMetadata("")
            h, v = _get_tile_hv(h5_path, meta)
            west, north = (10 * h) - 180, 90 - (10 * v)
            east, south = west + 10, north - 10

            # ---------- Build VRT ----------
            trans_opt = gdal.TranslateOptions(
                format="VRT",
                outputBounds=[west, north, east, south],
                outputSRS="EPSG:4326",
            )
            vrt_light = f"/vsimem/{h5_path.stem}_light.vrt"
            vrt_qual  = f"/vsimem/{h5_path.stem}_qual.vrt"
            vrt_cloud = f"/vsimem/{h5_path.stem}_cloud.vrt"
            vrt_snow  = f"/vsimem/{h5_path.stem}_snow.vrt"

            gdal.Translate(vrt_light,  light_path, options=trans_opt)
            gdal.Translate(vrt_qual,   qual_path,  options=trans_opt)
            gdal.Translate(vrt_cloud,  cloud_path, options=trans_opt)
            gdal.Translate(vrt_snow,   snow_path,  options=trans_opt)

            # ---------- Warp & 过滤 ----------
            warp_opt = gdal.WarpOptions(
                format="MEM",
                outputType=gdal.GDT_Float32,
                cutlineDSName=cutline_shp,
                cropToCutline=True,
                dstNodata=dst_nodata,
            )
            light_arr = gdal.Warp("", vrt_light,  options=warp_opt).ReadAsArray()
            qual_arr  = gdal.Warp("", vrt_qual,   options=warp_opt).ReadAsArray()
            cloud_arr = gdal.Warp("", vrt_cloud,  options=warp_opt).ReadAsArray()
            snow_arr  = gdal.Warp("", vrt_snow,   options=warp_opt).ReadAsArray()

            cloud_conf = (cloud_arr.astype(int) >> 6) & 0b11
            valid_mask = (qual_arr == 0) & (cloud_conf <= 1) & (snow_arr == 0)

            light_arr = light_arr.astype(np.float32) * 0.1
            light_arr[~valid_mask] = dst_nodata

            ref_ds = gdal.Warp("", vrt_light, options=warp_opt)
            xsize, ysize = ref_ds.RasterXSize, ref_ds.RasterYSize

            drv = gdal.GetDriverByName("GTiff")
            out_ds = drv.Create(str(out_tif), xsize, ysize, 1, gdal.GDT_Float32)
            out_ds.GetRasterBand(1).WriteArray(light_arr)
            out_ds.GetRasterBand(1).SetNoDataValue(dst_nodata)
            out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
            out_ds.SetProjection(ref_ds.GetProjection())
            out_ds = None

            tif_written.append(str(out_tif))
            # print(f"  ✓ 输出 {out_tif.name}")

        except Exception as err:
            print(f"  × 处理失败：{err}")

        finally:
            for vrt in (vrt_light, vrt_qual, vrt_cloud, vrt_snow):
                if vrt:
                    try:
                        gdal.Unlink(vrt)
                    except Exception:
                        pass

    return tif_written
