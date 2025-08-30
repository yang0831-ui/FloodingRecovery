import arcpy
import os

# 启用 Spatial Analyst 扩展
if arcpy.CheckExtension("Spatial") == "Available":
    arcpy.CheckOutExtension("Spatial")
else:
    raise RuntimeError("Spatial Analyst extension is not available.")

# 参数路径
tile_filter = "h09v06"

input_features = r"E:\National University of Singapore\Yang Yang - flooding\Other Place\12071.shp"#用于裁剪的shp文件
raster_folder = r"E:\National University of Singapore\Yang Yang - flooding\Process Data\h09v06_Florida\Geotiff\2018"
output_clipped_folder = r"E:\National University of Singapore\Yang Yang - flooding\Process Data\h09v06_Florida\Geotiff\2018\clipped"

if not os.path.exists(output_clipped_folder):
    os.makedirs(output_clipped_folder)

# 获取 shapefile 的 bounding box
desc = arcpy.Describe(input_features)
extent = desc.extent
bbox = f"{extent.XMin} {extent.YMin} {extent.XMax} {extent.YMax}"

# 裁剪所有 tif 文件到 H3 的 bounding box
for f in os.listdir(raster_folder):
    if f.endswith(".tif") and tile_filter in f:
        in_raster = os.path.join(raster_folder, f)
        out_raster = os.path.join(output_clipped_folder, f.replace(".tif", "_clip.tif"))

        if os.path.exists(out_raster):
            print(f"跳过已裁剪：{f}")
            continue

        try:
            arcpy.Clip_management(
                in_raster=in_raster,
                rectangle=bbox,
                out_raster=out_raster,
                in_template_dataset="",
                nodata_value="0",
                clipping_geometry="NONE",  # 用矩形裁剪
                maintain_clipping_extent="NO_MAINTAIN_EXTENT"
            )
            print(f"裁剪完成：{f}")
        except Exception as e:
            print(f"裁剪失败：{f}, 错误: {e}")