import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve, auc
import datetime

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

# Create Decision Tree model object
dt = DecisionTreeClassifier(random_state=1)

# Define the parameter grid for GridSearchCV
param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

# Create GridSearchCV object
grid_search = GridSearchCV(dt, param_grid=param_grid, cv=5)

# Perform Grid Search for hyperparameter tuning
grid_search.fit(X_train_std, y_train)

# Get the best model
best_dt_classifier = grid_search.best_estimator_

# Make predictions on the test set
y_pred_test = best_dt_classifier.predict(X_test_std)
y_pred_prob_test = best_dt_classifier.predict_proba(X_test_std)

# Calculate performance metrics and ROC curve

accuracy_test = accuracy_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test, average='weighted')
precision_test = precision_score(y_test, y_pred_test, average='weighted')

fpr_test = dict()
tpr_test = dict()
roc_auc_test = dict()

for i in range(best_dt_classifier.classes_.size):
    fpr_test[i], tpr_test[i], _ = roc_curve(y_test, y_pred_prob_test[:, i], pos_label=i)
    roc_auc_test[i] = auc(fpr_test[i], tpr_test[i])

# Print results and plot ROC curve
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_test)
print("Recall:", recall_test)
print("Precision:", precision_test)

plt.figure()
colors = ['blue', 'red', 'green']
for i, color in zip(range(best_dt_classifier.classes_.size), colors):
    plt.plot(fpr_test[i], tpr_test[i], color=color, lw=2, label='ROC curve (area = %0.2f) for class %s' % (roc_auc_test[i], i))
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Test Set')
plt.legend(loc="lower right")

current_time = datetime.datetime.now()
filename = current_time.strftime("%Y%m%d%H%M%S") + ".png"
save_path = "D:/Analysis of diabetes/Test_5/ROC/" + "Decision_Tree_Grid_Search_" + filename
plt.savefig(save_path)

plt.show()

# 将Accuracy, Precision, Recall保存到指定Excel中
ID = "Decision_Tree_grid_"
result_df = pd.DataFrame({'ID': [ID],'Accuracy': [accuracy_test], 'Precision': [precision_test], 'Recall': [recall_test],
                            'C':[grid_search.best_estimator_], 'save_path': [save_path]})
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