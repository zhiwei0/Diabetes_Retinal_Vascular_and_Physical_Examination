# # 将单因素回归分析和多因素回归分析的结果对应行合并，要求同指标名的才能合并，数据进行拼接
# import pandas as pd
#
# def merge_data_by_id(input_file1, input_file2, output_file):
#     # 读取文件1中的表1和文件2中的表2
#     xls = pd.ExcelFile(input_file1)
#     sheets = xls.sheet_names
#
#     with pd.ExcelWriter(output_file) as writer:
#         for sheet_name in sheets:
#             df1 = pd.read_excel(input_file1, sheet_name=sheet_name)
#             df2 = pd.read_excel(input_file2, sheet_name=sheet_name)
#
#             # 根据ID和NO列进行匹配，并拼接成新的数据行
#             merged_data = pd.merge(df1, df2, left_on=df1.columns[0], right_on=df2.columns[0], suffixes=('_file1', '_file2'))
#
#             # 将结果保存到新的表格中
#             merged_data.to_excel(writer, sheet_name=sheet_name, index=False)
#
# if __name__ == '__main__':
#     input_file1 = 'D:/Analysis of diabetes/Test_5/result_univariate_high_light.xlsx'
#     input_file2 = 'D:/Analysis of diabetes/Test_5/result_multivariate_high_light.xlsx'
#     output_file = 'D:/Analysis of diabetes/Test_5/result_merge.xlsx'   # 替换为输出结果的路径
#     merge_data_by_id(input_file1, input_file2, output_file)
import pandas as pd

def merge_data_by_id(input_file1, input_file2, output_file):
    # 读取文件1中的表1和文件2中的表2
    xls = pd.ExcelFile(input_file1)
    sheets = xls.sheet_names

    with pd.ExcelWriter(output_file) as writer:
        for sheet_name in sheets:
            df1 = pd.read_excel(input_file1, sheet_name=sheet_name)
            df2 = pd.read_excel(input_file2, sheet_name=sheet_name)

            # 根据ID进行匹配，并拼接成新的数据行，匹配的拼接在后面
            merged_data = pd.merge(df1, df2, left_on=df1.columns[0], right_on=df2.columns[0], how='outer')

            # 将结果保存到新的表格中
            merged_data.to_excel(writer, sheet_name=sheet_name, index=False)

if __name__ == '__main__':
    input_file1 = 'D:/Analysis of diabetes/Test_5/result_univariate_high_light.xlsx'
    input_file2 = 'D:/Analysis of diabetes/Test_5/result_multivariate_high_light.xlsx'
    output_file = 'D:/Analysis of diabetes/Test_5/result_merge.xlsx'   # 替换为输出结果的路径
    merge_data_by_id(input_file1, input_file2, output_file)
