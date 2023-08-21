import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import datetime as dt
import numpy as np

# Load data
file_path = "D:/Analysis of diabetes/Test_5/data_normalization_删除尿糖.xlsx"
sheet_name='正常范围_中间范围'
# sheet_name='中间范围_糖尿病'
diabetes = pd.read_excel(file_path,sheet_name=sheet_name)

# Prepare X and y
# 眼底+体检 第一次实验 0-1 多因素回归分析筛选出的指标
X = diabetes[['artery_Branching_Angle', 'BMI',
              '收缩压', '白蛋白(Alb)', '球蛋白(GLB)', '甘油三酯（TG）']]
# 眼底+体检 第一次实验 1-2 多因素回归分析筛选出的指标
# X = diabetes[['舒张压', '白蛋白(Alb)', '高密度脂蛋白胆固醇']]

y = diabetes['Result'].values.ravel()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Scale the features
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Create a Support Vector Regression (7_SVR) model
svr = SVR()

param_distributions = {
    'kernel': ['LR', 'poly', 'rbf', 'sigmoid'],  # Kernel type
    'C': np.logspace(-3, 3, 7),  # Regularization parameter
    'epsilon': np.logspace(-3, 1, 5)  # Epsilon parameter in the epsilon-7_SVR model
}

random_search = RandomizedSearchCV(estimator=svr, param_distributions=param_distributions, n_iter=50, cv=5)
random_search.fit(X_train_std, y_train)
y_pred_test = random_search.predict(X_test_std)

# Calculate mean squared error and R-squared for the test set
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

# Print the results for the test set
print("Best Parameters:", random_search.best_params_)
print("Mean Squared Error:", mse_test)
print("R-squared:", r2_test)

# Save the plot with a filename based on the current time
current_time = dt.datetime.now()
filename = current_time.strftime("%Y%m%d%H%M%S") + ".png"
save_path = "D:/Analysis of diabetes/Test_5/ROC/" + "SVR_Random_Search_" + filename
plt.savefig(save_path)

plt.show()

# 将Accuracy, Precision, Recall保存到指定Excel中
ID = "SVG_Randomized_"
result_df = pd.DataFrame({'ID': [ID],'Mean Squared Error': [mse_test], 'R-squared': [r2_test],
                            'C':[random_search.best_estimator_], 'save_path': [save_path]})
excel_filename = "D:/Analysis of diabetes/Test_5/预测模型结果.xlsx"

# 尝试读取已有的Excel文件
try:
    existing_df = pd.read_excel(excel_filename)
    result_df = pd.concat([existing_df,result_df],ignore_index=True)
except FileNotFoundError:
    print("未找到目标Excel文件")

# 保存到目标文件
with pd.ExcelWriter(excel_filename, mode='w', engine='openpyxl') as writer:
    result_df.to_excel(writer ,sheet_name='Sheet1', index=False)