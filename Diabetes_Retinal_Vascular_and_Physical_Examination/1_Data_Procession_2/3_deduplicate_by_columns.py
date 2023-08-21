# 第三步 将合并后的数据更加若干列为主键进行去重
# 根据指定的若干列进行去重操作
import pandas as pd

def deduplicate_by_columns(input_file_path, cols_1, cols_2, cols_3):
    # 读取整个Excel文件中的所有表格
    xls = pd.ExcelFile(input_file_path)
    sheets = xls.sheet_names

    # 遍历每个表格进行去重
    with pd.ExcelWriter(output_file) as writer:
        for sheet_name in sheets:
            df = pd.read_excel(xls, sheet_name=sheet_name)

            # 组合指定的若干列，进行去重操作
            deduplicated_data = df.drop_duplicates(subset=cols_1+cols_2+cols_3)

            # 将去重后的数据写回到原来的表格中
            deduplicated_data.to_excel(writer, sheet_name=sheet_name, index=False)

if __name__ == '__main__':
    input_file_path = 'D:/Analysis of diabetes/Test_5/data_merge.xlsx'
    output_file = 'D:/Analysis of diabetes/Test_5/data_filter1.xlsx'
    cols_1 = ['患者编号']  # 替换为需要用于去重的ID列名，可以是多个列，例如：['ID', 'Category']
    cols_2 = ['artery_caliber']  # 替换为需要用于去重的frac列名，可以是多个列，例如：['frac', 'Amount']
    cols_3 = ['vein_caliber']  # 替换为需要用于去重的XM列名，可以是多个列，例如：['XM', 'Date']
    deduplicate_by_columns(input_file_path, cols_1, cols_2, cols_3)
