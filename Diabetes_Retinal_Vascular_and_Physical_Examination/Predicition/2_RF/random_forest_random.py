import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV, train_test_split
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
# rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=1)
# rf.fit(X_train_std, y_train)
rf = RandomForestClassifier()

# 定义随机搜索的参数空间
param_dist = {
    'n_estimators': randint(1, 1000),           # 决策树的数量范围在10到200之间随机选择
    'max_depth': randint(1, 1000),                # 最大深度范围在1到20之间随机选择
    'min_samples_split': randint(1, 1000),        # 内部节点分裂所需的最小样本数范围在2到20之间随机选择
    'min_samples_leaf': randint(1, 1000),         # 叶节点所需的最小样本数范围在1到20之间随机选择
    'max_features': ['auto', 'sqrt', 'log2'],   # 最佳分割特征数从三个选项中随机选择
    'bootstrap': [True, False],                # 是否使用有放回抽样随机选择
    'class_weight': [None, 'balanced']         # 类别权重从两个选项中随机选择
}

# 创建随机搜索对象
random_search = RandomizedSearchCV(
    rf, param_distributions=param_dist, n_iter=50, cv=5, random_state=None, n_jobs=None)

# 在训练数据上进行随机搜索调参
random_search.fit(X_train, y_train)

y_pred = random_search.predict(X_test_std)
y_pred_prob = random_search.predict_proba(X_test_std)

# 在测试集上进行预测
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted',zero_division='warn')
recall = recall_score(y_test, y_pred, average='weighted')

# 输出结果
print("Best Parameters:", random_search.best_estimator_)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# 计算并绘制ROC曲线
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(random_search.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_prob[:, i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
colors = ['blue', 'red', 'green']
for i, color in zip(range(len(random_search.classes_)), colors):
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
save_path = "D:/Analysis of diabetes/Test_5/ROC/" +"random_forest_" + filename
plt.savefig(save_path)

plt.show()

# 将Accuracy, Precision, Recall保存到指定Excel中
ID = "Random_Forest_random_"
result_df = pd.DataFrame({'ID': [ID],'Accuracy': [accuracy], 'Precision': [precision], 'Recall': [recall],
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
