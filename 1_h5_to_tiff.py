# 选择非gis解释器运行

from osgeo import gdal
import os
inputFolder = r'E:\National University of Singapore\Yang Yang - flooding\Raw Data\California\2018'
outputFolder = r'E:\National University of Singapore\Yang Yang - flooding\Geotiff\California\2018_with_cloud'
os.makedirs(outputFolder, exist_ok=True)  # 保证输出目录存在
os.chdir(inputFolder)
# 获取所有 .h5 或 .hdf 文件
rasterFiles = [f for f in os.listdir(os.getcwd()) if f.endswith('.h5') or f.endswith('.hdf')]
for fname in rasterFiles:
    print(f"处理文件：{fname}")
    rasterFilePre = fname[:-3]
    fileExtension = "_BBOX.tif"
    hdflayer = gdal.Open(fname, gdal.GA_ReadOnly)
    if hdflayer is None:
        print(f"无法打开 {fname}，跳过。")
        continue
    subdatasets = hdflayer.GetSubDatasets()
    if not subdatasets:
        print(f"{fname} 没有子数据集，跳过。")
        continue
    # 取第一个波段（夜间灯光主数据层）
    subhdflayer = subdatasets[0][0]
    rlayer = gdal.Open(subhdflayer, gdal.GA_ReadOnly)
    if rlayer is None:
        print(f"无法打开子数据集 {subhdflayer}，跳过。")
        continue
    # 生成输出文件名
    outputName = subhdflayer[92:]
    outputNameNoSpace = outputName.strip().replace(" ", "_").replace("/", "_")
    outputNameFinal = outputNameNoSpace + rasterFilePre + fileExtension
    outputRaster = os.path.join(outputFolder, outputNameFinal)
    print("输出路径:", outputRaster)
# 计算地理范围
    meta = rlayer.GetMetadata_Dict()
    try:
        HorizontalTileNumber = int(meta["HorizontalTileNumber"])
        VerticalTileNumber = int(meta["VerticalTileNumber"])
        WestBoundCoord = (10 * HorizontalTileNumber) - 180
        NorthBoundCoord = 90 - (10 * VerticalTileNumber)
        EastBoundCoord = WestBoundCoord + 10
        SouthBoundCoord = NorthBoundCoord - 10
    except Exception as e:
        print(f"读取元数据失败：{e}，跳过。")
        continue
    EPSG = "-a_srs EPSG:4326"
    translateOptionText = f"{EPSG} -a_ullr {WestBoundCoord} {NorthBoundCoord} {EastBoundCoord} {SouthBoundCoord}"
    translateoptions = gdal.TranslateOptions(gdal.ParseCommandLine(translateOptionText))
    # 执行转换
    gdal.Translate(outputRaster, rlayer, options=translateoptions)
    print(f"{fname} 转换完成！\n")
print("全部批量转换完成！")