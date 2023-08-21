# 第五步 由于LDR的两列0值较多，故将这两列删除
# 删除LDR的两列数据
import pandas as pd

def delete_by_columns(input_file_path, output_file_path, cols_1):
    # 读取整个Excel文件中的所有表格
    xls = pd.ExcelFile(input_file_path)
    sheets = xls.sheet_names


    with pd.ExcelWriter(output_file_path) as writer:
        for sheet_name in sheets:
            df = xls.parse(sheet_name)

            df.drop(columns = cols_1, inplace = True, errors = 'ignore')

            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print("处理完成并保存为新的Excel文件。")

if __name__ == '__main__':
    input_file_path = 'D:/Analysis of diabetes/Test_5/data_filter1.xlsx'
    output_file_path = 'D:/Analysis of diabetes/Test_5/data_delete_LDR.xlsx'
    cols_1 = ['artery_LDR','vein_LDR','血糖','糖化血红蛋白']  # 替换为需要删除的ID列名，可以是多个列，例如：['ID', 'Category']
    # cols_2 = ['vein_LDR']
    # cols_3 = ['vein_caliber']
    delete_by_columns(input_file_path, output_file_path, cols_1)


