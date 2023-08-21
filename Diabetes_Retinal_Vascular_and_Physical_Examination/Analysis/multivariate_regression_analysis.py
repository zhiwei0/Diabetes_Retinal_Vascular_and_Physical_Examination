import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
# 读取Excel表格

df = pd.read_excel("D:/Analysis of diabetes/Test_5/data_normalization1.xlsx",sheet_name='正常范围_中间范围')
# 只能使用下标
X = df.iloc[:, [16,29,30,31,34,35,36,37,38,39,40,44,46,49]]
y = df.iloc[:, 1]


# 添加截距项
X = sm.add_constant(X)

# 多因素逻辑回归模型拟合
model = sm.Logit(y, X)
results = model.fit()
params = results.params
conf_int = results.conf_int()
or_vals = np.exp(params)
or_vals.name = "OR"
conf_int = np.exp(conf_int)
p_values = results.pvalues
p_values.name = "P值"

# 合并所有结果到一个DataFrame中，并按每个自变量一行的形式输出
summary = pd.concat([or_vals, conf_int, p_values], axis=1)
summary.columns = ["OR", "95% CI Lower", "95% CI Upper", "P值"]
summary.index = X.columns

# Exclude the 'const' row from the summary
summary_without_const = summary.iloc[1:]

# Export the final results to an Excel file
output_file = "D:/Analysis of diabetes/Test_5/multivariate_results_0_1.xlsx"
summary_without_const.to_excel(output_file)
print("Results exported to Excel successfully.")
