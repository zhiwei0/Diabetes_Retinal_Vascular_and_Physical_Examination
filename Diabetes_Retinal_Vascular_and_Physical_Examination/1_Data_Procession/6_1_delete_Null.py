import pandas as pd
import os

def process_excel_file(input_file_path, output_file_path):
    # 读取Excel文件中的所有表格
    xls = pd.ExcelFile(input_file_path)
    sheet_names = xls.sheet_names

    # # 创建一个空的DataFrame来保存合并后的结果
    # merged_df = pd.DataFrame()

    # 处理每张表格
    for idx, sheet_name in enumerate(sheet_names):

        # 读取当前表格
        df = xls.parse(sheet_name)

        # 删除指定若干列，可以将需要删除的列名放在这个列表中
        # columns_to_delete = ['XM']
        # df.drop(columns=columns_to_delete, inplace=True, errors='ignore')

        # 删除包含0的数据行
        df = df[~(df == 0).any(axis=1)]

        # 删除存在空值的数据行
        df.dropna(inplace=True)

        # 合并当前表格到merged_df中
        # merged_df = pd.concat([merged_df, df], ignore_index=True)

        # 保存为新的Excel文件，每张表格保存在同一个文件中，表格名为sheet_name
        # 保存合并后的结果到新的Excel文件
        if not os.path.exists(output_file_path):
            with pd.ExcelWriter(output_file_path, engine='openpyxl', mode='w') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            with pd.ExcelWriter(output_file_path, engine='openpyxl', mode='a') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)

# 调用函数处理Excel文件，假设输入文件为"input_file.xlsx"，输出文件为"output_file.xlsx"
input_file_path = "D:/Analysis of diabetes/Test_5/data_delete_LDR.xlsx"
output_file_path  = "D:/Analysis of diabetes/Test_5/data_删除空值行.xlsx"

process_excel_file(input_file_path, output_file_path)
