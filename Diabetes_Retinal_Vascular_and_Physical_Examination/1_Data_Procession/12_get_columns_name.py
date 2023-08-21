import pandas as pd
# 此处获取待归一化的指标来赋值归一化中的cols_to_normalize值
# 读取Excel文件，指定文件路径和sheet_name参数（如果sheet在多个工作表中有相同的名称，可以使用sheet_name参数指定具体的sheet）
file_path = 'D:/Analysis of diabetes/Test_5/data_normalization.xlsx'
df = pd.read_excel(file_path)

# 获取列名列表
column_names = df.columns.tolist()

# 格式化输出列名
for column in column_names:
    print(column)
