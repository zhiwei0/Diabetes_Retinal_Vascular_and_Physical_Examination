import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve, auc
import datetime as dt

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

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# Scale the features
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Create an 10_XGBoost classifier
xgb = XGBClassifier(random_state=1)

param_dist = {
    'n_estimators': randint(50, 201),  # 在[50, 200]范围内随机选择整数作为n_estimators的值
    'learning_rate': [0.01, 0.1, 0.2],  # Step size at each iteration
}

# n_iter参数指定了随机搜索要执行的随机采样次数，可以根据需要设置为合适的值
random_search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=20, cv=5, random_state=1)

random_search.fit(X_train_std, y_train)

best_xgb_classifier = random_search.best_estimator_
best_xgb_classifier.fit(X_train_std, y_train)

# 进行预测
y_pred_test = best_xgb_classifier.predict(X_test_std)
y_pred_prob_test = best_xgb_classifier.predict_proba(X_test_std)

# 计算性能指标
accuracy_test = accuracy_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test, average='weighted')
precision_test = precision_score(y_test, y_pred_test, average='weighted')

# 计算ROC曲线和AUC
fpr_test = dict()
tpr_test = dict()
roc_auc_test = dict()

for i in range(best_xgb_classifier.classes_.size):
    fpr_test[i], tpr_test[i], _ = roc_curve(y_test, y_pred_prob_test[:, i], pos_label=i)
    roc_auc_test[i] = auc(fpr_test[i], tpr_test[i])

# Print the results for test set
print("Best Parameters:", random_search.best_params_)
print("Accuracy:", accuracy_test)
print("Recall:", recall_test)
print("Precision:", precision_test)


# Plotting the ROC curves for test set
plt.figure()
colors = ['blue', 'red', 'green']
for i, color in zip(range(best_xgb_classifier.classes_.size), colors):
    plt.plot(fpr_test[i], tpr_test[i], color=color, lw=2, label='ROC curve (area = %0.2f) for class %s' % (roc_auc_test[i], i))
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Test Set')
plt.legend(loc="lower right")

# 将图像根据时间命名并保存到指定路径
current_time = dt.datetime.now()
filename = current_time.strftime("%Y%m%d%H%M%S") + ".png"
save_path = "D:/Analysis of diabetes/Test_5/ROC/" + "XGBoost_Random_Search_" + filename
plt.savefig(save_path)

plt.show()

# 将Accuracy, Precision, Recall保存到指定Excel中
ID = "XGBoost_random_"
result_df = pd.DataFrame({'ID': [ID],'Accuracy': [accuracy_test], 'Precision': [precision_test], 'Recall': [recall_test],
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