import pandas as pd
import os

# 定义处理函数
def process_table(df, columns_to_check):
    for col in columns_to_check:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # 将非数值数据转换为NaN
    df = df.dropna(subset=columns_to_check)  # 删除含有NaN的行
    return df

# 输入文件路径和输出文件路径
input_file_path = "D:/Analysis of diabetes/Test_5/data_删除空值行.xlsx"
output_file_path = "D:/Analysis of diabetes/Test_5/data_删除非数值.xlsx"

# 指定要检查的列
cols_to_check = ['年龄', '性别', '舒张压', '收缩压',
                 '总蛋白(TP)', '白蛋白(Alb)', '球蛋白(GLB)', '总胆红素(T-BIL)',
                 '直接胆红素(DB)', '间接胆红素(IB)', '谷丙转氨酶(ALT)', '谷草转氨酶(AST)',
                 '尿素氮(BUN)', '肌酐(Cr)', '尿酸(UA)', '总胆固醇（TC）', '甘油三酯（TG）',
                 '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇']  # 需要检查的列名

# 读取Excel文件中的每个表格，进行处理并保存到新的Excel文件
with pd.ExcelWriter(output_file_path) as writer:
    for sheet_name in pd.ExcelFile(input_file_path).sheet_names:
        df = pd.read_excel(input_file_path, sheet_name=sheet_name)
        df_processed = process_table(df, cols_to_check)
        df_processed.to_excel(writer, sheet_name=sheet_name, index=False)

print("处理完成并已保存到新的Excel文件。")