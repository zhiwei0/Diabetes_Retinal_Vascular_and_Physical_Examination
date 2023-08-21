import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import datetime as dt

# Load data
# diabetes = pd.read_excel("D:/Analysis of diabetes/data_random_select.xlsx",sheet_name='正常范围_中间范围')
# diabetes = pd.read_excel("D:/Analysis of diabetes/data_random_select.xlsx",sheet_name='中间范围_糖尿病')

diabetes = pd.read_excel("D:/Analysis of diabetes/data_process_normalization.xlsx",sheet_name='正常范围_中间范围')
# diabetes = pd.read_excel("D:/Analysis of diabetes/data_process_normalization.xlsx",sheet_name='中间范围_糖尿病')

# diabetes = pd.read_excel("D:/Analysis of diabetes/Test_2/data_process_normalization.xlsx",sheet_name='正常范围_中间范围')
# diabetes = pd.read_excel("D:/Analysis of diabetes/Test_2/data_process_normalization.xlsx",sheet_name='中间范围_糖尿病')

# Prepare X and y
# 0-1 筛选出的指标
# X = diabetes[['frac','vein_Num1stBa','artery_Branching_Angle','artery_Asymmetry_Raito']]
# 1-2 筛选出的指标
# X = diabetes[['frac','vein_curvature','artery_Num1stBa','artery_Branching_Angle','vein_Branching_Angle','artery_Angle_Asymmetry','artery_Optimal_Deviation','vessel_length_density','vessel_area_density']]
# 1-2 多因素回归的指标
# X = diabetes[['frac','vein_Branching_Angle','artery_Optimal_Deviation','vessel_length_density']]
# 所有指标
X = diabetes[['artery_caliber','vein_caliber','frac','AVR','artery_curvature','vein_curvature','artery_BSTD','vein_BSTD',
              'artery_simple_curvatures','vein_simple_curvatures','artery_Branching_Coefficient','vein_Branching_Coefficient',
              'artery_Num1stBa','vein_Num1stBa','artery_Branching_Angle','vein_Branching_Angle','artery_Angle_Asymmetry',
              'vein_Angle_Asymmetry','artery_LDR','vein_LDR','artery_Asymmetry_Raito','vein_Asymmetry_Raito','artery_JED',
              'vein_JED','artery_Optimal_Deviation','vein_Optimal_Deviation','vessel_length_density','vessel_area_density']]

y = diabetes['Result'].values.ravel()
# 进行 1-0 2-1 转化
# y = diabetes['Result'].replace({1: 0,2: 1}).values.ravel()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Standardize data
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Fit logistic regression model
lr = LogisticRegression(C=1.873928814379866, random_state=1, multi_class='ovr', solver='lbfgs', max_iter=10000)

# Train model
lr.fit(X_train_std, y_train)

# Predict classes and probabilities for test data
y_pred = lr.predict(X_test_std)
y_pred_prob = lr.predict_proba(X_test_std)

# Calculate accuracy, recall, and precision
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
print("Class:",lr.classes_)
print("Coef:",lr.coef_)
print("intercept",lr.intercept_)
print("n_iter",lr.n_iter_)

print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)

# Calculate ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(lr.classes_.size):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_prob[:, i], pos_label=i)
    # fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_prob[:, 1], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
colors = ['blue', 'red', 'green']
for i, color in zip(range(lr.classes_.size), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve (area = %0.2f) for class %s' % (roc_auc[i], i))
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")

# 将图像根据时间命名并保存到指定路径
current_time = dt.datetime.now()
filename =  current_time.strftime("%Y%m%d%H%M%S") + ".png"
save_path = "D:/Analysis of diabetes/Test_4/ROC/" + filename
plt.savefig(save_path)

plt.show()
