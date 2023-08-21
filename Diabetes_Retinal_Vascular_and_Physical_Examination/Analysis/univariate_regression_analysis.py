import pandas as pd
import numpy as np
import scipy.optimize
import statsmodels.api as sm

# 读取Excel文件
#1 使用特征归一化后的数据
file_path = "D:/Analysis of diabetes/Test_5/data_normalization1.xlsx"
df = pd.read_excel(file_path,sheet_name='中间范围_糖尿病')

# 拆分数据为自变量和因变量
medical_data = df.loc[:, df.columns[2:52]]

# Initialize an empty DataFrame to store the final results
results_df = pd.DataFrame()

# Iterate through each column of medical_data
for col in medical_data.columns:
    # 获取目标变量与预测变量
    X = medical_data[col]
    y=df.iloc[:, 1]

    # 添加常数项截距
    X = sm.add_constant(X)

    # 构建逻辑回归模型
    model = sm.Logit(y, X)
    result = model.fit()
    params = result.params
    conf_int = result.conf_int()
    or_vals = np.exp(params)
    or_vals.name = "OR"
    conf_int = np.exp(conf_int)
    p_values = result.pvalues
    p_values.name = "P值"

    # 合并所有结果到一个DataFrame中，并按每个自变量一行的形式输出
    summary = pd.concat([or_vals, conf_int, p_values], axis=1)
    summary.columns = ["OR", "95% CI Lower", "95% CI Upper", "P值"]
    summary.index = X.columns

    print(summary)

    # Exclude the 'const' row from the summary
    summary_without_const = summary.iloc[1:]

    # Add the current summary to the final results DataFrame
    results_df = pd.concat([results_df, summary_without_const], axis=0)

# Export the final results to an Excel file
output_file = "D:/Analysis of diabetes/Test_5/result_univariate.xlsx"
results_df.to_excel(output_file)
print("Results exported to Excel successfully.")
