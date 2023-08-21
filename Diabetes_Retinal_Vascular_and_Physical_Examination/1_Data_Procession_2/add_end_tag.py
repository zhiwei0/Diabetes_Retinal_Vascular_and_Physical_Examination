import pandas as pd


def add_suffix_to_first_row(input_file_path, output_file_path):
    # 读取Excel文件
    df = pd.read_excel(input_file_path)

    # 为第一行（标题行）数据添加后缀“_R1”
    df.columns = [col + '_L2' for col in df.columns]

    # 保存修改后的数据到原Excel文件
    df.to_excel(output_file_path, index=False)


input_file_path = "D:/Analysis of diabetes/Test_5/指标加后缀.xlsx"
output_file_path = "D:/Analysis of diabetes/Test_5/指标加后缀_1.xlsx"
add_suffix_to_first_row(input_file_path, output_file_path)
