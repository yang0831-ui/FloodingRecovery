import arcpy
import pandas as pd
import os

# 设置输出表格所在目录
input_folder = r"E:\National University of Singapore\Yang Yang - flooding\Process Data\h09v06_Florida\dbf\2017xy"
output_csv = r"E:\National University of Singapore\Yang Yang - flooding\Process Data\h09v06_Florida\dbf\2017xy\summary_wide.csv"

# 获取所有 .dbf 文件
dbf_files = [f for f in os.listdir(input_folder) if f.endswith(".dbf")]

merged_df = pd.DataFrame()

for dbf in dbf_files:
    dbf_path = os.path.join(input_folder, dbf)

    try:
        # 提取时间标签（可选）
        label = os.path.splitext(dbf)[0]  # 如 A2016153_h09v06

        # 转换为 numpy array，再变成 DataFrame
        array = arcpy.da.TableToNumPyArray(dbf_path, "*")
        df = pd.DataFrame(array)

        # 添加新列用于标识来源文件
        df["filename"] = label

        merged_df = pd.concat([merged_df, df], ignore_index=True)
        print(f"已合并：{dbf}")

    except Exception as e:
        print(f"跳过：{dbf}，错误：{e}")

# 提取日期信息
merged_df['year'] = merged_df['filename'].str[1:5]
merged_df['day_of_year'] = merged_df['filename'].str[5:].astype(int)
merged_df['date'] = pd.to_datetime(merged_df['year'] + merged_df['day_of_year'].astype(str), format='%Y%j')
merged_df = merged_df.drop(columns=['year', 'day_of_year'])

# 转为宽格式
merged_df_wide = merged_df.pivot(index='xy_id', columns='date', values='MEAN')
merged_df_wide = merged_df_wide.reset_index().rename(columns={'index': 'xy_id'})

# 保存为 CSV
merged_df_wide.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"\n全部完成，已导出：{output_csv}")

# 删去dbf文件

suffixes = (".dbf", ".cpg", ".dbf.xml") 

for dbf in dbf_files:                    
    stem = os.path.splitext(dbf)[0]
    for suf in suffixes:
        filepath = os.path.join(input_folder, stem + suf)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"已删除：{filepath}")
            except Exception as e:
                print(f"无法删除 {filepath}：{e}")

print("\n清理完毕。")