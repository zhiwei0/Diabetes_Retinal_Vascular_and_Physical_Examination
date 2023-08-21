import pandas as pd

# 指定要读取的Excel文件路径
excel_path = 'D:/Analysis of diabetes/Test_5/result_multivariate_high_light.xlsx'

# 读取Excel文件中的所有表名
xls = pd.ExcelFile(excel_path)
sheet_names = xls.sheet_names

# 指定要获取的列名，例如"Name"
column_name = "ID"

# 遍历每张表
for sheet_name in sheet_names:
    # 读取每张表
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # 获取指定列的所有值
    column_values = df[column_name].tolist()

    # # 打印或使用列中的每一行
    # for value in column_values:
    #     formatted_value = f"'{value}'"
    #     print(f"{formatted_value}")
    print(column_values)
