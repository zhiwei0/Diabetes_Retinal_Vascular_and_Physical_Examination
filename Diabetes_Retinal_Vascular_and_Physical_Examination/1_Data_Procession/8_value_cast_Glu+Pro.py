# 第七步 对于定性指标需要进行量化
import pandas as pd

# 读取Excel文件，指定文件路径和sheet_name参数（如果sheet在多个工作表中有相同的名称，可以使用sheet_name参数指定具体的sheet）
file_path = 'D:/Analysis of diabetes/Test_5/data_filter2.xlsx'
output_file = "D:/Analysis of diabetes/Test_5/data_value_GLU+Pro_cast.xlsx"

df_dict = pd.read_excel(file_path, sheet_name=None)

# 针对每张表格，指定要替换的列和替换的映射关系
columns_to_replace = {'尿糖（Glu）': {'阴性': 0, '+-': 1, '1+': 1, '2+': 1, '3+': 1, '4+': 1},
                      '蛋白PRO': {'阴性': 0, '+-': 1, '1+': 1, '2+': 1, '3+': 1, '4+': 1}}

# 处理每张表格
with pd.ExcelWriter(output_file) as writer:
    for sheet_name, df in df_dict.items():
        # 针对每列进行值替换
        for column, replace_dict in columns_to_replace.items():
            df[column] = df[column].replace(replace_dict)

        # 保存到新的Excel文件
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print("处理完成并保存为新的Excel文件。")
