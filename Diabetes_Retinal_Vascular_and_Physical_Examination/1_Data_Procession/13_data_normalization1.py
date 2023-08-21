# 第十步 对数据进行特征归一化
import pandas as pd

#读取 Excel 文件
# 此处读取的文件应该是预处理阶段倒数第二步的数据，直接可用于本步归一化
file_path = 'D:/Analysis of diabetes/Test_5/data_ready_0.xlsx'

df_0_1 = pd.read_excel(file_path,sheet_name='正常范围_中间范围')
df_1_2 = pd.read_excel(file_path,sheet_name='中间范围_糖尿病')

#获取原值减去均值除以方差的特征列
cols_to_normalize = ['artery_caliber', 'vein_caliber', 'frac', 'AVR', 'artery_curvature',
                    'vein_curvature', 'artery_BSTD', 'vein_BSTD', 'artery_simple_curvatures',
                    'vein_simple_curvatures', 'artery_Branching_Coefficient', 'vein_Branching_Coefficient',
                    'artery_Num1stBa', 'vein_Num1stBa', 'artery_Branching_Angle', 'vein_Branching_Angle',
                    'artery_Angle_Asymmetry', 'vein_Angle_Asymmetry', 'artery_Asymmetry_Raito',
                    'vein_Asymmetry_Raito', 'artery_JED', 'vein_JED', 'artery_Optimal_Deviation',
                    'vein_Optimal_Deviation', 'vessel_length_density', 'vessel_area_density',
                    '年龄', '性别', 'BMI', '舒张压', '收缩压', '尿糖（Glu）', '蛋白PRO', '总蛋白(TP)',
                    '白蛋白(Alb)', '球蛋白(GLB)', '总胆红素(T-BIL)', '直接胆红素(DB)', '间接胆红素(IB)',
                    '谷丙转氨酶(ALT)', '谷草转氨酶(AST)', '尿素氮(BUN)', '肌酐(Cr)', '尿酸(UA)', '总胆固醇（TC）',
                    '甘油三酯（TG）', '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇']

df_0_1[cols_to_normalize] = df_0_1[cols_to_normalize].apply(lambda x:(x - x.mean()) / x.std())
df_1_2[cols_to_normalize] = df_1_2[cols_to_normalize].apply(lambda x:(x - x.mean()) / x.std())

#保存修改后的数据列至新的 Excel 文件
new_file_path = "D:/Analysis of diabetes/Test_5/data_normalization.xlsx"
with pd.ExcelWriter(new_file_path) as writer:
    df_0_1.to_excel(writer,sheet_name='正常范围_中间范围',index=False)
    df_1_2.to_excel(writer,sheet_name='中间范围_糖尿病',index=False)

print("特征归一化处理并保存为新的Excel文件。")


