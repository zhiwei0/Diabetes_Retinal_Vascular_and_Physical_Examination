# 第八步 把三张表合并为0-1、1-2
import pandas as pd

# 读取Excel文件，指定文件路径和sheet_name参数
# file_path = 'D:/Analysis of diabetes/Test_5/data_value_cast.xlsx'
# output_file = "D:/Analysis of diabetes/Test_5/data_process1.xlsx"
file_path = 'D:/Analysis of diabetes/Test_5/data_add_BMI.xlsx'
output_file = "D:/Analysis of diabetes/Test_5/data_physical_combine.xlsx"

df_dict = pd.read_excel(file_path, sheet_name=None)

# 合并sheet1和sheet2为sheet01
sheet01 = pd.concat([df_dict['正常范围'], df_dict['中间范围']])

# 合并sheet2和sheet3为sheet12
sheet12 = pd.concat([df_dict['中间范围'], df_dict['糖尿病']])

# 保存到新的Excel文件
with pd.ExcelWriter(output_file) as writer:
    sheet01.to_excel(writer, sheet_name='正常范围_中间范围', index=False)
    sheet12.to_excel(writer, sheet_name='中间范围_糖尿病', index=False)

print("合并并保存为新的Excel文件。")
