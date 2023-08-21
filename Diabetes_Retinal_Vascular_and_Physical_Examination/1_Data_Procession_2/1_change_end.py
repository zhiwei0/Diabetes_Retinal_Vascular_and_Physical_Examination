# 第一步
# 表格中ID中数据格式为xxxx_xxxx.png，新增NO为只保留下划线前面的字段，即患者编号
import pandas as pd

def modify_column_id(input_file_path):
    # 读取整个Excel文件中的所有表格
    xls = pd.ExcelFile(input_file_path)
    sheets = xls.sheet_names

    # 遍历每个表格，对指定列ID中的内容进行修改
    with pd.ExcelWriter(output_file_path) as writer:
        for sheet_name in sheets:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            if 'ID' in df.columns:
                df['NO'] = df['ID'].str.split('_').str[0]

                # 将修改后的DataFrame写回到原始表格中
                df.to_excel(writer, sheet_name=sheet_name, index=False)

if __name__ == '__main__':
    input_file_path = 'D:/Analysis of diabetes/Test_5/data.xlsx'  # 输入Excel文件路径
    output_file_path = 'D:/Analysis of diabetes/Test_5/data_split_end.xlsx'
    modify_column_id(input_file_path)