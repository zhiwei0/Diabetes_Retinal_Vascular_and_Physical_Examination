import prediction
import pandas as pd
from itertools import chain, combinations

def all_subsets(arr):
    return chain(*map(lambda x: combinations(arr, x), range(2, len(arr) + 1)))


def Pred1():
    input_file_path = "D:/Analysis of diabetes/Test_8/data_normalization.xlsx"
    sheet_names = ['正常范围_中间范围', '中间范围_糖尿病']
    output_file_path = "D:/Analysis of diabetes/Test_8/预测模型结果.xlsx"
    pic_path = "D:/Analysis of diabetes/Test_8/ROC/"
    cols_y = 'Result'
    cols_x = ['artery_caliber', 'vein_caliber', 'frac', 'AVR', 'artery_curvature', 'vein_curvature', 'artery_BSTD', 'vein_BSTD', 'artery_simple_curvatures', 'vein_simple_curvatures', 'artery_Branching_Coefficient', 'vein_Branching_Coefficient', 'artery_Num1stBa', 'vein_Num1stBa', 'artery_Branching_Angle', 'vein_Branching_Angle', 'artery_Angle_Asymmetry', 'vein_Angle_Asymmetry', 'artery_Asymmetry_Raito', 'vein_Asymmetry_Raito', 'artery_JED', 'vein_JED', 'artery_Optimal_Deviation', 'vein_Optimal_Deviation', 'vessel_length_density', 'vessel_area_density']

    for sheet_name in sheet_names:
        prediction.Prediction_Models(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y)


if __name__ == '__main__':
    Pred1()




