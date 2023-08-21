# 第一步
# 把体检数据所有数据中Result等于0、1、2的数据分别复制到原文件的一张新表格中
# 输入的Excel中只有一张表，包含了正常、中间、糖尿病三种状态的数据
# 数据处理的步骤：
# 1 data_physical_examination.xlsx
# 2 data_retinal_vascular.xlsx
# 3 data_merge.xlsx
# 4 data_filter1
# 5 data_delete_LDR.xlsx
# 6 data_filter2.xlsx
# 7 data_value_cast.xlsx
# 8 data_process1.xlsx
# 9 data_process1.xlsx
# 10 data_normalization1.xlsx
import pandas as pd

def copy_data_by_result(input_file_path):
    # 读取整个Excel文件中的所有表格
    xls = pd.ExcelFile(input_file_path)
    sheets = xls.sheet_names

    # 为每个Result值创建一个空的DataFrame
    result_dfs = {0: pd.DataFrame(), 1: pd.DataFrame(), 2: pd.DataFrame()}

    # 遍历每个表格，根据Result值拆分数据并复制到相应的DataFrame中
    df = pd.read_excel(xls, sheet_name='Sheet1')
    for result_value in [0, 1, 2]:
        result_df = df[df['Result'] == result_value]
        result_dfs[result_value] = pd.concat([result_dfs[result_value], result_df])

    # 打开输入的Excel文件，并将每个Result值对应的DataFrame写入到不同表格中
    with pd.ExcelWriter(input_file_path, mode='a', engine='openpyxl') as writer:
        for result_value, result_df in result_dfs.items():
            result_df.to_excel(writer, sheet_name='Sheet{}'.format(result_value), index=False)

if __name__ == '__main__':
    input_file_path = 'D:/Analysis of diabetes/Test_5/data_physical_examination.xlsx'  # 输入Excel文件路径
    copy_data_by_result(input_file_path)
