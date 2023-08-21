# 第九步 对与合并后的两张表，将1-2中Result换成0-1后续就不用进行1-0，2-1的转换，这样每张表的result都只取0/1
import pandas as pd

# 读取Excel文件，指定文件路径和sheet_name参数
file_path = 'D:/Analysis of diabetes/Test_5/data_physical_combine.xlsx'
sheet_name = '中间范围_糖尿病'  # 指定要处理的表格名称


# 全局的替换规则字典，可以使用嵌套字典或者字典列表
# 使用嵌套字典的例子：
# 进行数值替换时无需加''
column_replace_dict = {
    'Result': {1: 0, 2: 1}
}

# 或者使用字典列表的例子：
# column_replace_dict = [
#     {'column': 'Column1', 'mapping': {'a': 1, 'b': 2, 'c': 3, 'd': 4}},
#     {'column': 'Column2', 'mapping': {'x': 'X', 'y': 'Y', 'z': 'Z'}}
# ]

# 读取指定表格数据
df = pd.read_excel(file_path, sheet_name=sheet_name)

# 针对每列进行值替换（使用全局的替换规则字典）
for column, value_map in column_replace_dict.items():
    if column in df.columns:
        df[column] = df[column].replace(value_map)

# 保存修改后的数据到原文件中
with pd.ExcelWriter(file_path, mode='a', if_sheet_exists='replace') as writer:
    df.to_excel(writer, sheet_name=sheet_name, index=False)

print("值替换并保存到原Excel文件。")
