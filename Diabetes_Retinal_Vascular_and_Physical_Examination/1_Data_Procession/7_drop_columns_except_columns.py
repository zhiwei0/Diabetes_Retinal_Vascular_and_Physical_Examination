# 第六步 把包含0的无意义的数据行删除
# Result、年龄、性别中取值为0是有意义的，不能删除，否则将至少有一半的数据被删掉
import pandas as pd

# 读取Excel文件，指定文件路径和sheet_name参数（如果sheet在多个工作表中有相同的名称，可以使用sheet_name参数指定具体的sheet）
file_path = "D:/Analysis of diabetes/Test_5/data_删除非数值.xlsx"
output_file = "D:/Analysis of diabetes/Test_5/data_filter2.xlsx"


df_dict = pd.read_excel(file_path, sheet_name=None)

# 需要保留的列名（即除Result列以外的列名）
columns_to_keep = ['ID', 'artery_caliber', 'vein_caliber', 'frac', 'AVR', 'artery_curvature',
                    'vein_curvature', 'artery_BSTD', 'vein_BSTD', 'artery_simple_curvatures',
                    'vein_simple_curvatures', 'artery_Branching_Coefficient', 'vein_Branching_Coefficient',
                    'artery_Num1stBa', 'vein_Num1stBa', 'artery_Branching_Angle', 'vein_Branching_Angle',
                    'artery_Angle_Asymmetry', 'vein_Angle_Asymmetry', 'artery_Asymmetry_Raito',
                    'vein_Asymmetry_Raito', 'artery_JED', 'vein_JED', 'artery_Optimal_Deviation',
                    'vein_Optimal_Deviation', 'vessel_length_density', 'vessel_area_density',
                    '体重', '身高', '舒张压', '收缩压',  '总蛋白(TP)',
                    '白蛋白(Alb)', '球蛋白(GLB)', '总胆红素(T-BIL)', '直接胆红素(DB)', '间接胆红素(IB)',
                    '谷丙转氨酶(ALT)', '谷草转氨酶(AST)', '尿素氮(BUN)', '肌酐(Cr)', '尿酸(UA)', '总胆固醇（TC）',
                    '甘油三酯（TG）', '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇']

with pd.ExcelWriter(output_file) as writer:
# 处理每张表格
    for sheet_name, df in df_dict.items():
        # 选取除Result列以外的列
        df_to_process = df[columns_to_keep]

        # 删除包含0的行（不包括Result列）
        df_to_process = df_to_process[~(df_to_process == 0).any(axis=1)]

        # 将Result列添加回DataFrame
        # df_to_process['Result'] = df['Result']
        df_to_process['年龄'] = df['年龄']
        df_to_process['性别'] = df['性别']
        df_to_process['尿糖（Glu）'] = df['尿糖（Glu）']
        df_to_process['蛋白PRO'] = df['蛋白PRO']

        df_to_process.dropna(inplace = True)

        # 保存到新的Excel文件
        df_to_process.to_excel(writer, sheet_name=sheet_name, index=False)

print("处理完成并保存为新的Excel文件。")

