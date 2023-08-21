from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# Load data
file_path = "D:/Analysis of diabetes/Test_5/data_normalization.xlsx"
sheet_name='正常范围_中间范围'
# sheet_name='中间范围_糖尿病'
diabetes = pd.read_excel(file_path,sheet_name=sheet_name)

# Prepare X and y
# 眼底+体检 第一次实验 0-1 多因素回归分析筛选出的指标
X = diabetes[['artery_simple_curvatures_L1', 'artery_Optimal_Deviation_L1']]
# 眼底+体检 第一次实验 1-2 多因素回归分析筛选出的指标
# X = diabetes[['vein_Optimal_Deviation_R2']]

y = diabetes['Result'].values.ravel()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)

# Standardize data
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# 创建逻辑回归模型
# lr = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)
# lr = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000)
# lr = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
# lr = LogisticRegression(penalty='l2', solver='newton-cg', max_iter=1000)
lr = LogisticRegression(penalty='l2', solver='saga', max_iter=1000)

# 设置超参数候选值
param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.8, 1, 2, 4, 8, 10, 20, 16, 32, 100, 1000, 10000]}

# 创建网格搜索对象
grid_search = GridSearchCV(lr, param_grid, cv=5, n_jobs=-1)

# 在训练集上进行网格搜索
grid_search.fit(X_train, y_train)

# 使用最优参数的模型进行预测
best_lr = grid_search.best_estimator_
y_pred = best_lr.predict(X_test_std)
y_pred_prob = grid_search.predict_proba(X_test_std)

# 在测试集上进行预测
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

#输出最优的C值
print("Best C:", grid_search.best_params_['C'])
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# 计算并绘制ROC曲线
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(grid_search.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_prob[:, i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
colors = ['blue', 'red', 'green']
for i, color in zip(range(len(grid_search.classes_)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve (area = %0.2f) for class %s' % (roc_auc[i], i))
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)-Testing set')
plt.legend(loc="lower right")

# 将图像根据时间命名并保存到指定路径
current_time = dt.datetime.now()
filename =  current_time.strftime("%Y%m%d%H%M%S") + ".png"
save_path = "D:/Analysis of diabetes/Test_5/ROC/" +"LR_" + filename
plt.savefig(save_path)
plt.show()

# 将Accuracy, Precision, Recall保存到指定Excel中
ID = "liblinear_grid_"+grid_search.best_estimator_.solver
result_df = pd.DataFrame({'ID': [ID],'Accuracy': [accuracy], 'Precision': [precision], 'Recall': [recall],
                          'C':[grid_search.best_params_['C']], 'save_path': [save_path]})
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
