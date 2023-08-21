import pandas as pd

# 输入文件路径和输出文件路径
input_file_path = 'D:/Analysis of diabetes/Test_5/data_value_GLU+Pro_cast.xlsx'
output_file_path = 'D:/Analysis of diabetes/Test_5/data_add_BMI.xlsx'

# 定义计算BMI的函数
def calculate_bmi(df):
    df['身高（m）'] = df['身高'] / 100  # 将厘米转换为米
    df['BMI'] = df['体重'] / (df['身高（m）'] * df['身高（m）'])

    # 删除指定若干列，可以将需要删除的列名放在这个列表中
    columns_to_delete = ['身高（m）','身高','体重']
    df.drop(columns=columns_to_delete, inplace=True, errors='ignore')
    return df

# 读取Excel文件中的每个表格，进行BMI计算并保存到新的Excel文件
with pd.ExcelWriter(output_file_path) as writer:
    for sheet_name in pd.ExcelFile(input_file_path).sheet_names:
        df = pd.read_excel(input_file_path, sheet_name=sheet_name)
        df = calculate_bmi(df)
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print("BMI计算完成并已保存到新的Excel文件。")
