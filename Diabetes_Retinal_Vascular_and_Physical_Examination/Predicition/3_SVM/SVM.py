# 不自动调参的SVM
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

# # Create an 3_SVM classifier
# svm = SVC(C=1.0, kernel='rbf', random_state=1, probability=True)
# svm.fit(X_train_std, y_train)
#
# # Perform predictions on the test set
# y_pred = svm.predict(X_test_std)
# y_pred_prob = svm.predict_proba(X_test_std) , probability=True
# svm = SVC(random_state=1 , probability=True)
# param_grid = {
#     'C':[1.0,2.0,None],
#     'kernel':['rbf','1_LR','poly'],
#     'degree':[3,6,9],
#     'coef0':[0.0,1.0,2.0]
# }
# grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5)
# grid_search.fit(X_train_std, y_train)
# y_pred = grid_search.predict(X_test_std)
# y_pred_prob = grid_search.predict_proba(X_test_std)
svm = SVC(C=3.0, kernel='rbf',degree=3,coef0=0.0, random_state=1, probability=True)
svm.fit(X_train_std, y_train)

y_pred_train = svm.predict(X_train_std)
y_pred_prob_train = svm.predict_proba(X_train_std)
accuracy_train = accuracy_score(y_train, y_pred_train)
recall_train = recall_score(y_train, y_pred_train, average='weighted')
precision_train = precision_score(y_train, y_pred_train, average='weighted')
fpr_train = dict()
tpr_train = dict()
roc_auc_train = dict()

for i in range(svm.classes_.size):
    fpr_train[i], tpr_train[i], _ = roc_curve(y_train, y_pred_prob_train[:, i], pos_label=i)
    roc_auc_train[i] = auc(fpr_train[i], tpr_train[i])

print("Accuracy:", accuracy_train)
print("Recall:", recall_train)
print("Precision:", precision_train)

# Plotting the ROC curves for training set
plt.figure()
colors = ['blue', 'red', 'green']
for i, color in zip(range(svm.classes_.size), colors):
    plt.plot(fpr_train[i], tpr_train[i], color=color, lw=2, label='ROC curve (area = %0.2f) for class %s' % (roc_auc_train[i], i))
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Training Set')
plt.legend(loc="lower right")

# 将图像根据时间命名并保存到指定路径
current_time = dt.datetime.now()
filename =  current_time.strftime("%Y%m%d%H%M%S") + ".png"
save_path = "D:/Analysis of diabetes/Test_5/ROC/" +"SVM_" + filename
plt.savefig(save_path)

plt.show()