import pandas as pd

# 定义函数进行数据拼接操作
def combine_rows(df):
    combined_data = pd.DataFrame()

    for i in range(0, len(df), 4):
        id_group = df.iloc[i:i + 4]

        # 使用`join`方法将这四行数据拼接成一个大行
        combined_row = id_group.values.flatten()

        # 添加拼接后的数据到新的DataFrame中
        combined_data = combined_data.append([combined_row], ignore_index=True)

    return combined_data


# 定义函数处理Excel文件中的每张表格
def process_excel_sheets(input_file_path, output_file_path):
    xls = pd.ExcelFile(input_file_path)
    sheets = xls.sheet_names

    # 创建新的Excel文件
    with pd.ExcelWriter(output_file_path) as writer:
        for sheet_name in sheets:
            # 读取Excel文件
            sheet_data = pd.read_excel(input_file_path, sheet_name=sheet_name)

            # 对每张表格进行数据拼接操作
            combined_data = combine_rows(sheet_data)

            # 输出结果
            print(f"Combined data for sheet '{sheet_name}'")
            # print(combined_data)

            # 将结果写入新的Excel文件，指定sheet名称为原表格名称
            combined_data.to_excel(writer, sheet_name=sheet_name, index=False)

if __name__ == '__main__':
    input_file_path = 'D:/Analysis of diabetes/Test_5/data1.xlsx'  # 替换为您的输入Excel文件路径
    output_file_path = 'D:/Analysis of diabetes/Test_5/data_connect.xlsx'  # 替换为您的输出文件目录路径
    process_excel_sheets(input_file_path, output_file_path)
