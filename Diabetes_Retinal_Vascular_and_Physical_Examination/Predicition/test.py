import numpy as np
import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# Load data
diabetes = pd.read_excel("D:/Analysis of diabetes/data_process_normalization.xlsx",sheet_name='中间范围_糖尿病')

# Prepare X and y
# X = diabetes[['NL', 'XB','BMI','收缩压','尿糖（Glu）','蛋白PRO','谷丙转氨酶(ALT)','谷草转氨酶(AST)','甘油三酯（TG）']]
# y = diabetes['result'].replace({1: 0, 2: 1})
X = diabetes[['frac','vein_Branching_Angle','artery_Optimal_Deviation','vessel_length_density']]
y = diabetes['Result'].replace({1:0,2:1})

# # Replace "<3.00" and "<3" with NaN, and fill NaNs with the means
# X.replace({'<3.00': np.nan, '<3': np.nan}, inplace=True)
# X.fillna(X.mean(), inplace=True)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Standardize data
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Fit logistic regression model
lr = LogisticRegression(C=358.47799527463786, random_state=1, multi_class='multinomial', solver='lbfgs', max_iter=10000)
lr.fit(X_train_std, y_train)


# Predict classes and probabilities for test data
y_pred = lr.predict(X_test_std)
y_pred_prob = lr.predict_proba(X_test_std)
# y_pred_train = lr.predict(X_train_std)
# y_pred_prob_train = lr.predict_proba(X_train_std)
# Calculate accuracy, recall, and precision


# accuracy_train = accuracy_score(y_train, y_pred_train)
# recall_train = recall_score(y_train, y_pred_train, average='weighted')
# precision_train = precision_score(y_train, y_pred_train, average='weighted')
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
print("Class:",lr.classes_)
print("Coef:",lr.coef_)
print("intercept",lr.intercept_)
print("n_iter",lr.n_iter_)
# ## 预测前三样本在各个类别的概率
# print("前五样本在各个类别的预测概率为：\n",lr.predict_proba(X_test_std[:5,:]))
# print("\n============================")
# ## 获得前三个样本的分类标签
# print("\n前五样本在各个类别的预测类别为：\n",lr.predict(X_test_std[:5,:]))
# print("\n============================")
# print("\n前五样本在各个类别的真实类别为：\n",y_test[:5])
# print("\n============================")
print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)

# Calculate ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(lr.classes_.size):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_prob[:, i], pos_label=i)
    # fpr[i], tpr[i], _ = roc_curve(y_train, X_train_std[:, i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(lr.classes_.size):
    fpr[i], tpr[i], _ = roc_curve(y_test.ravel(), y_pred_prob[:, i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])
# Plot ROC curves
plt.figure()
colors = ['blue', 'red', 'green']
for i, color in zip(range(lr.classes_.size), colors):
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
save_path = "D:/Analysis of diabetes/ROC/" + filename
plt.savefig(save_path)

# 显示图像
plt.show()

# fpr_train = dict()
# tpr_train = dict()
# roc_auc_train = dict()
#
# for i in range(lr.classes_.size):
#     fpr_train[i], tpr_train[i], _ = roc_curve(y_train, y_pred_prob_train[:, i], pos_label=i)
#     roc_auc_train[i] = auc(fpr_train[i], tpr_train[i])
#
# print("Training Set Metrics:")
# print("Accuracy:", accuracy_train)
# print("Recall:", recall_train)
# print("Precision:", precision_train)
# print("Class:", lr.classes_)
# print("Coef:", lr.coef_)
# print("Intercept:", lr.intercept_)
# print("n_iter:", lr.n_iter_)
# print("前五样本在各个类别的预测概率为：\n", lr.predict_proba(X_train_std[:5, :]))
# print("前五样本在各个类别的预测类别为：\n", lr.predict(X_train_std[:5, :]))
#
# # Plotting the ROC curves for training set
# plt.figure()
# colors = ['blue', 'red', 'green']
# for i, color in zip(range(lr.classes_.size), colors):
#     plt.plot(fpr_train[i], tpr_train[i], color=color, lw=2, label='ROC curve (area = %0.2f) for class %s' % (roc_auc_train[i], i))
# plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) - Training Set')
# plt.legend(loc="lower right")
# plt.show()


# Calculate and print multi-class AUC
# print("Multi-class AUC:", roc_auc_score(y_test, y_pred_prob, multi_class='ovr'))
# auc_value = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
# print("AUC:", auc_value)
# auc_value = roc_auc_score(y_test.ravel(), y_pred_prob, multi_class='ovr')
# print("AUC:", auc_value)