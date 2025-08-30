import arcpy
import os
import re
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support

# === 1. 路径参数 ===
input_folder  = r"E:\National University of Singapore\Yang Yang - flooding\Process Data\h09v06_Florida\Geotiff\2017\clipped"
output_folder = r"E:\National University of Singapore\Yang Yang - flooding\Process Data\h09v06_Florida\dbf\2017xy"
fishnet_shp   = r"E:\National University of Singapore\Yang Yang - flooding\Process Data\h09v06_Florida\pixel_fishnet.shp"
zone_field    = "index"            # 生成网格后会创建

# === 2. 文件命名函数（与原来一致） ===
def sanitize_name(raw_name):
    m = re.search(r"\.A(\d{7})\.", raw_name)
    date7 = m.group(1) if m else re.sub(r'[^0-9]', '', raw_name)[:7].ljust(7, '_')
    return f"d{date7}.dbf"

# === 3. 仅首次运行时，创建像素鱼网格 ===
def add_xy_id_field(fishnet_shp):
    # 如果已经有 xy_id 字段，则跳过
    fields = [f.name for f in arcpy.ListFields(fishnet_shp)]
    if "xy_id" in fields:
        print("字段 xy_id 已存在，跳过创建。")
        return

    # 新增字段
    arcpy.management.AddField(fishnet_shp, "xy_id", "TEXT", field_length=50)

    # 用中心点生成 xy_id
    with arcpy.da.UpdateCursor(fishnet_shp, ["SHAPE@XY", "xy_id"]) as cursor:
        for row in cursor:
            x, y = row[0]          # row[0] 是 (x, y) 元组
            lon_dir = "W" if x < 0 else "E"
            lat_dir = "S" if y < 0 else "N"
            xy_id   = f"{lon_dir}{abs(x):.5f}{lat_dir}{abs(y):.5f}"
            row[1]  = xy_id
            cursor.updateRow(row)


def build_pixel_fishnet(sample_raster, out_shp):
    if os.path.exists(out_shp):
        return  # 已有则跳过

    # 1) 读取模板影像的范围和分辨率
    ras = arcpy.Raster(sample_raster)
    env = arcpy.env
    env.workspace   = os.path.dirname(sample_raster)
    env.snapRaster  = ras              # 保证对齐
    env.extent      = ras.extent
    env.outputCoordinateSystem = ras.spatialReference

    cell_w = ras.meanCellWidth
    cell_h = ras.meanCellHeight
    xmin, ymin, xmax, ymax = ras.extent.XMin, ras.extent.YMin, ras.extent.XMax, ras.extent.YMax

    # 2) CreateFishnet。origin＝左下角，y_axis=“{xmin} {ymin+1}”即可
    arcpy.management.CreateFishnet(
        out_feature_class = out_shp,
        origin_coord      = f"{xmin} {ymin}",
        y_axis_coord      = f"{xmin} {ymin+1}",
        cell_width        = cell_w,
        cell_height       = cell_h,
        number_rows       = "0",   # 让工具自动按行列数生成
        number_columns    = "0",
        corner_coord      = f"{xmax} {ymax}",
        labels            = "NO_LABELS",
        template          = sample_raster,
        geometry_type     = "POLYGON"
    )

    # 添加 index 字段
    arcpy.management.AddField(out_shp, "index", "LONG")
    arcpy.management.CalculateField(out_shp, "index", "!FID!", "PYTHON3")

    # 立刻写入 xy_id，避免第二次获取 schema 锁
    arcpy.management.AddField(out_shp, "xy_id", "TEXT", field_length=50)
    with arcpy.da.UpdateCursor(out_shp, ["SHAPE@XY", "xy_id"]) as cur:
        for row in cur:
            x, y = row[0]
            xy_id = f"{'W' if x<0 else 'E'}{abs(x):.5f}{'S' if y<0 else 'N'}{abs(y):.5f}"
            row[1] = xy_id
            cur.updateRow(row)

# === 4. 单文件处理函数 ===
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
            in_zone_data   = fishnet_shp,
            zone_field     = 'xy_id',
            in_value_raster= file_path,
            out_table      = output_table,
            ignore_nodata  = "DATA",
            statistics_type= "MEAN"      # 每个像元自身的值
        )
        return f"完成：{table_name}"
    except Exception as e:
        return f"失败：{table_name}，错误：{e}"

# === 5. 主程序 ===
def main():
    if arcpy.CheckExtension("Spatial") != "Available":
        raise RuntimeError("Spatial Analyst extension is not available.")
    arcpy.CheckOutExtension("Spatial")

    os.makedirs(output_folder, exist_ok=True)

    # 5-1 先创建像素网格（取第一幅影像作为模板）
    sample_ras = next((os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith("_clip.tif")), None)
    if not sample_ras:
        raise RuntimeError("输入文件夹中未找到 *_clip.tif 文件。")
    build_pixel_fishnet(sample_ras, fishnet_shp)



    # 5-2 并行跑所有影像
    tif_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith("_clip.tif")]
    with ProcessPoolExecutor(max_workers=4) as executor:
        for result in executor.map(process_zonal, tif_files):
            print(result)

    print("\n全部处理完成。")

if __name__ == '__main__':
    freeze_support()
    main()
