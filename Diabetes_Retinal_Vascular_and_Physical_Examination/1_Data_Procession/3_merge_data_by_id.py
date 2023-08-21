# 第三步 将体检数据和眼底血管数据合并，要求同患者ID的才能合并，数据进行拼接
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
            merged_data = pd.merge(df1, df2, left_on='ID', right_on='患者编号', suffixes=('_file1', '_file2'))

            # 删除'患者编号'和'患者姓名'列
            merged_data.drop(['患者编号', 'XM'], axis=1, inplace=True)

            # 将'Result'列移到第二列
            # 此处Result先不急移动，会导致最后一列消失
            # cols = merged_data.columns.tolist()
            # cols = [cols[0]] + ['Result'] + cols[1:-1]
            # merged_data = merged_data[cols]

            # 将结果保存到新的表格中
            merged_data.to_excel(writer, sheet_name=sheet_name, index=False)

if __name__ == '__main__':
    input_file1 = 'D:/Analysis of diabetes/Test_5/data_retinal_vascular_end.xlsx'
    input_file2 = 'D:/Analysis of diabetes/Test_5/data_physical_examination.xlsx'
    output_file = 'D:/Analysis of diabetes/Test_5/data_merge.xlsx'   # 替换为输出结果的路径
    merge_data_by_id(input_file1, input_file2, output_file)


