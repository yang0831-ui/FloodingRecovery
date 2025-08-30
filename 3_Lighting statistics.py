import arcpy
import os
import re
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support

# shared globals
h3_shapefile = r"E:\National University of Singapore\Yang Yang - flooding\新建文件夹\H3_8th_6097.shp"
zone_field = "index"
input_folder = r"E:\National University of Singapore\Yang Yang - flooding\Geotiff\California\2019N_perfect_clipped——8th"
output_folder = r"E:\National University of Singapore\Yang Yang - flooding\h05v05_county06097_2019N_perfect_clipped——8th"

def sanitize_name(raw_name):

    m = re.search(r"\.A(\d{7})\.", raw_name)
    if m:
        date7 = m.group(1)
    else:
        # 前 7 个字符保险
        date7 = re.sub(r'[^0-9]', '', raw_name)[:7].ljust(7, '_')
    return f"d{date7}.dbf"

def process_zonal(file_path):

    arcpy.CheckOutExtension("Spatial")

    fname      = os.path.basename(file_path)
    raw_name   = fname.replace("_clip.tif", "")
    table_name = sanitize_name(raw_name)
    output_table = os.path.join(output_folder, table_name)

    if os.path.exists(output_table):
        return f"已存在：{table_name}"

    try:
        arcpy.sa.ZonalStatisticsAsTable(
            in_zone_data=h3_shapefile,
            zone_field=zone_field,
            in_value_raster=file_path,
            out_table=output_table,
            ignore_nodata="DATA",
            statistics_type="MEAN"
        )
        return f"完成：{table_name}"
    except Exception as e:
        return f"失败：{table_name}，错误：{e}"

def main():

    if arcpy.CheckExtension("Spatial") == "Available":
        arcpy.CheckOutExtension("Spatial")
    else:
        raise RuntimeError("Spatial Analyst extension is not available.")

    # 新建文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 找到所有已裁剪的 tif 文件
    tif_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.endswith("_clip.tif")
    ]

    with ProcessPoolExecutor(max_workers=4) as executor:
        for result in executor.map(process_zonal, tif_files):
            print(result)

    print("\n全部处理完成。")

if __name__ == '__main__':
    freeze_support()
    main()
