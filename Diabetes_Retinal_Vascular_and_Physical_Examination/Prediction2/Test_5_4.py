import pandas as pd
# 眼底血管数据四个角度单独作为指标，进行所有-单因素-多因素三次实验。
from Analysis_and_Prediction import Prediction_Models
from Prediction2.Test_8 import add_value_to_excel_column1

# 获取列名
def get_cols_name():
    input_file_path = "D:/Analysis of diabetes/Test_5/data_normalization.xlsx"
    df = pd.read_excel(input_file_path)
    columns_name = df.columns.tolist()
    print(columns_name)


# 所有指标代入
def prediction():
    input_file_path = "D:/Analysis of diabetes/Test_5/data_normalization.xlsx"
    sheet_names = ['正常范围_中间范围', '中间范围_糖尿病']
    output_file_path = "D:/Analysis of diabetes/Test_5/预测模型结果_所有指标.xlsx"
    pic_path = "D:/Analysis of diabetes/Test_5/ROC/"
    cols_x = ['artery_caliber_R1', 'vein_caliber_R1', 'frac_R1', 'AVR_R1', 'artery_curvature_R1', 'vein_curvature_R1', 'artery_BSTD_R1', 'vein_BSTD_R1', 'artery_simple_curvatures_R1', 'vein_simple_curvatures_R1', 'artery_Branching_Coefficient_R1', 'vein_Branching_Coefficient_R1', 'artery_Num1stBa_R1', 'vein_Num1stBa_R1', 'artery_Branching_Angle_R1', 'vein_Branching_Angle_R1', 'artery_Angle_Asymmetry_R1', 'vein_Angle_Asymmetry_R1', 'artery_Asymmetry_Raito_R1', 'vein_Asymmetry_Raito_R1', 'artery_JED_R1', 'vein_JED_R1', 'artery_Optimal_Deviation_R1', 'vein_Optimal_Deviation_R1', 'vessel_length_density_R1', 'vessel_area_density_R1',
              'artery_caliber_L1', 'vein_caliber_L1', 'frac_L1', 'AVR_L1', 'artery_curvature_L1', 'vein_curvature_L1', 'artery_BSTD_L1', 'vein_BSTD_L1', 'artery_simple_curvatures_L1', 'vein_simple_curvatures_L1', 'artery_Branching_Coefficient_L1', 'vein_Branching_Coefficient_L1', 'artery_Num1stBa_L1', 'vein_Num1stBa_L1', 'artery_Branching_Angle_L1', 'vein_Branching_Angle_L1', 'artery_Angle_Asymmetry_L1', 'vein_Angle_Asymmetry_L1', 'artery_Asymmetry_Raito_L1', 'vein_Asymmetry_Raito_L1', 'artery_JED_L1', 'vein_JED_L1', 'artery_Optimal_Deviation_L1', 'vein_Optimal_Deviation_L1', 'vessel_length_density_L1', 'vessel_area_density_L1',
              'artery_caliber_R2', 'vein_caliber_R2', 'frac_R2', 'AVR_R2', 'artery_curvature_R2', 'vein_curvature_R2', 'artery_BSTD_R2', 'vein_BSTD_R2', 'artery_simple_curvatures_R2', 'vein_simple_curvatures_R2', 'artery_Branching_Coefficient_R2', 'vein_Branching_Coefficient_R2', 'artery_Num1stBa_R2', 'vein_Num1stBa_R2', 'artery_Branching_Angle_R2', 'vein_Branching_Angle_R2', 'artery_Angle_Asymmetry_R2', 'vein_Angle_Asymmetry_R2', 'artery_Asymmetry_Raito_R2', 'vein_Asymmetry_Raito_R2', 'artery_JED_R2', 'vein_JED_R2', 'artery_Optimal_Deviation_R2', 'vein_Optimal_Deviation_R2', 'vessel_length_density_R2', 'vessel_area_density_R2',
              'artery_caliber_L2', 'vein_caliber_L2', 'frac_L2', 'AVR_L2', 'artery_curvature_L2', 'vein_curvature_L2', 'artery_BSTD_L2', 'vein_BSTD_L2', 'artery_simple_curvatures_L2', 'vein_simple_curvatures_L2', 'artery_Branching_Coefficient_L2', 'vein_Branching_Coefficient_L2', 'artery_Num1stBa_L2', 'vein_Num1stBa_L2', 'artery_Branching_Angle_L2', 'vein_Branching_Angle_L2', 'artery_Angle_Asymmetry_L2', 'vein_Angle_Asymmetry_L2', 'artery_Asymmetry_Raito_L2', 'vein_Asymmetry_Raito_L2', 'artery_JED_L2', 'vein_JED_L2', 'artery_Optimal_Deviation_L2', 'vein_Optimal_Deviation_L2', 'vessel_length_density_L2', 'vessel_area_density_L2']
    cols_y = 'Result'

    for sheet_name in sheet_names:
        columns_values = {'cols_x': cols_x,
                          'sheet_name': sheet_name
                          }
        add_value_to_excel_column1(output_file_path, 'Sheet1', columns_values)

        Prediction_Models.Prediction_Models(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y)

    print("所有指标代入预测完成")


# 使用单因素回归分析筛出的指标
def prediction1():
    input_file_path = "D:/Analysis of diabetes/Test_5/data_normalization.xlsx"
    sheet_names = ['正常范围_中间范围', '中间范围_糖尿病']
    output_file_path = "D:/Analysis of diabetes/Test_5/预测模型结果_单因素.xlsx"
    pic_path = "D:/Analysis of diabetes/Test_5/ROC/"
    cols_x_01 = ['vein_Branching_Angle_R1', 'artery_simple_curvatures_L1', 'artery_Optimal_Deviation_L1', 'vein_Asymmetry_Raito_R2', 'artery_Asymmetry_Raito_L2']
    cols_x_12 = ['vessel_length_density_R1', 'vessel_area_density_R1', 'vein_Optimal_Deviation_R2', 'frac_L2', 'vessel_area_density_L2']
    cols = [cols_x_01, cols_x_12]
    cols_y = 'Result'

    for sheet_name, cols_x in zip(sheet_names, cols):

        columns_values = {
                          'sheet_name': sheet_name,
                          'cols_x': cols_x
                          }
        add_value_to_excel_column1(output_file_path, 'Sheet1', columns_values)

        Prediction_Models.Prediction_Models(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y)

    print("单因素回归分析指标代入预测完成")


# 使用多因素回归分析筛出的指标
def prediction2():
    input_file_path = "D:/Analysis of diabetes/Test_5/data_normalization.xlsx"
    sheet_names = ['正常范围_中间范围', '中间范围_糖尿病']
    output_file_path = "D:/Analysis of diabetes/Test_5/预测模型结果_多因素1.xlsx"
    pic_path = "D:/Analysis of diabetes/Test_5/ROC/"
    cols_x_01 = ['artery_simple_curvatures_L1', 'artery_Optimal_Deviation_L1']
    cols_x_12 = ['vein_Optimal_Deviation_R2']
    cols = [cols_x_01, cols_x_12]
    cols_y = 'Result'

    for sheet_name, cols_x in zip(sheet_names, cols):

        columns_values = {
                          'sheet_name': sheet_name,
                          'cols_x': cols_x
                          }
        add_value_to_excel_column1(output_file_path, 'Sheet1', columns_values)

        Prediction_Models.Prediction_Models(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y)

    print("多因素回归分析指标代入预测完成")


if __name__ == '__main__':
    # prediction()
    # prediction1()
    prediction2()
    # get_cols_name()