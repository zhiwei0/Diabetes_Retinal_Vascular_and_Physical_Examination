# 第二步 将体检数据和眼底血管数据合并，要求同患者ID的才能合并，进行筛选
import pandas as pd

def merge_data_by_id(input_file1, input_file2, output_file):
    # 读取文件1中的表1和文件2中的表2
    xls = pd.ExcelFile(input_file1)
    sheets = xls.sheet_names

    with pd.ExcelWriter(output_file) as writer:
        for sheet_name in sheets:
            df1 = pd.read_excel(input_file1, sheet_name=sheet_name)
            df2 = pd.read_excel(input_file2, sheet_name=sheet_name)

            # 根据ID和NO列进行匹配，并拼接成新的数据行
            merged_data = pd.merge(df1, df2, left_on='NO', right_on='患者编号', suffixes=('_file1', '_file2'))

            # 将结果保存到新的表格中
            merged_data.to_excel(writer, sheet_name=sheet_name, index=False)

if __name__ == '__main__':
    input_file1 = 'D:/Analysis of diabetes/Test_5/data_split_end.xlsx'
    input_file2 = 'D:/Analysis of diabetes/Test_5/data_physical_examination.xlsx'
    output_file = 'D:/Analysis of diabetes/Test_5/data_merge.xlsx'   # 替换为输出结果的路径
    merge_data_by_id(input_file1, input_file2, output_file)
