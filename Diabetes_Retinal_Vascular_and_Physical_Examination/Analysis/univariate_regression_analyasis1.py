# 可以循环执行单因素回归分析每一张表格
import pandas as pd
import numpy as np
import scipy.optimize
import statsmodels.api as sm

# 读取Excel文件
file_path = "D:/Analysis of diabetes/Test_5/data_normalization.xlsx"

# 获取Excel文件中的所有表格名
xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names

# 创建新的Excel文件来保存结果
output_file = "D:/Analysis of diabetes/Test_5/result_univariate.xlsx"

output_workbook = pd.ExcelWriter(output_file, engine='openpyxl')

# 遍历每张表格进行处理
for sheet_name in sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # 拆分数据为自变量和因变量
    # medical_data = df.loc[:, df.columns[2:27]]
    medical_data = df.loc[:, df.columns[2:106]]
    y = df.iloc[:, 1]

    # Initialize an empty DataFrame to store the results for the current sheet
    results_df = pd.DataFrame()

    # 遍历每一列进行单因素回归分析
    for col in medical_data.columns:
        # 获取目标变量与预测变量
        X = medical_data[col]

        # 添加常数项截距
        X = sm.add_constant(X)

        # 构建逻辑回归模型
        model = sm.Logit(y, X)
        result = model.fit()
        # result =  model.fit_regularized(alpha=0,method='l1')
        params = result.params
        conf_int = result.conf_int()
        or_vals = np.exp(params)
        or_vals.name = "OR"
        conf_int = np.exp(conf_int)
        p_values = result.pvalues
        p_values.name = "P值"

        # 合并当前列的结果到一个DataFrame中，并按每个自变量一行的形式输出
        summary = pd.concat([or_vals, conf_int, p_values], axis=1)
        summary.columns = ["OR", "95% CI Lower", "95% CI Upper", "P值"]
        summary.index = X.columns

        # Exclude the 'const' row from the summary
        summary_without_const = summary.iloc[1:]

        # 添加当前列的结果到当前表格结果DataFrame中
        results_df = pd.concat([results_df, summary_without_const], axis=0)

    # 将当前表格的结果保存到新的Excel文件的当前表中，并设置样式
    results_df.to_excel(output_workbook, sheet_name=sheet_name, index=True)

# 保存结果到新的Excel文件
output_workbook.save()

print("Results exported to Excel successfully.")
