from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd

input_file_path = "D:/Analysis of diabetes/Test_8/data_normalization.xlsx"
output_file_path = "D:/Analysis of diabetes/Test_8/预测结果.xlsx"
roc_save_path = "D:/Analysis of diabetes/Test_8/ROC/"
sheet_name1 = "sheet1"
sheet_names = ['正常范围_中间范围', '中间范围_糖尿病']

def plot_and_save_roc(model, X_test, y_test, model_name, roc_save_path):
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc="lower right")

    # 获取当前日期和时间
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")

    # 保存路径和文件名
    filename = f"{model_name.replace(' ', '_')}_{current_time}.png"
    full_save_path = roc_save_path + filename
    plt.savefig(full_save_path)

    plt.show()
    print(model_name+"的ROC已完成！")
    plt.close()

def save_metrics_to_excel(model_name, accuracy, precision, recall, selected_features, file_path):
    # 创建一个数据字典
    data = {
        'Model Name': [model_name],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'Features': [selected_features]
    }

    # 创建一个DataFrame
    df = pd.DataFrame(data)

    # 尝试读取已有的Excel文件
    try:
        existing_df = pd.read_excel(file_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        print("未找到目标Excel文件")

    # 保存到目标文件
    with pd.ExcelWriter(file_path, mode='w', engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)



def get_param_grid(model_name):
    # 定义模型特定的超参数网格
    param_grid = {}
    if model_name == 'Logistic Regression':
        param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.8, 1, 2, 4, 8, 10, 20, 16, 32, 100, 1000, 10000]}
    elif model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [3, 5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }
    elif model_name == 'Support Vector Machine':
        param_grid = {
            'base_estimator__C': [0.5, 1.0, 2.0, 3.0],
            'base_estimator__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'base_estimator__degree': [3, 6, 9],
            'base_estimator__coef0': [0.0, 1.0, 2.0]
        }
    elif model_name == 'Decision Tree':
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5]
        }
    elif model_name == 'K-Nearest Neighbors':
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]  # 1: Manhattan distance, 2: Euclidean distance
        }
    elif model_name == 'Neural Network':
        param_grid = {
            'hidden_layer_sizes': [(100,), (50, 50), (50, 25, 10)],  # Number of neurons in each hidden layer
            'activation': ['relu', 'tanh', 'logistic'],  # Activation function
            'alpha': [0.0001, 0.001, 0.01],  # L2 regularization parameter
            'max_iter': [100000]
        }
    elif model_name == 'Gradient Boosting Tree':
        param_grid = {
            'n_estimators': [50, 100, 200],  # Number of boosting stages to be run
            'learning_rate': [0.01, 0.1, 0.2],  # Step size at each iteration
            'max_depth': [3, 5, 7],  # Maximum depth of the individual trees
        }
    elif model_name == 'AdaBoost':
        param_grid = {
            'n_estimators': [50, 100, 200],  # Number of boosting stages to be run
            'learning_rate': [0.01, 0.1, 0.2],  # Step size at each iteration
        }
    elif model_name == 'XGBoost':
        param_grid = {
            'n_estimators': [50, 100, 200],  # Number of boosting stages to be run
            'learning_rate': [0.01, 0.1, 0.2],  # Step size at each iteration
            'max_depth': [3, 5, 7],  # Maximum depth of the individual trees
        }
    return param_grid

def auto_tuning(model, model_name, X_train, y_train):
    param_grid = get_param_grid(model_name)
    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               cv=5,
                               n_jobs=-1,
                               scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def train_and_evaluate_model(model_class, model_name, file_path, sheet_name, is_svc=False):
    # 读取数据
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    X = data.iloc[:, 2:]
    y = data['Result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 特征选择
    sfs = SequentialFeatureSelector(model_class(), n_features_to_select=5, direction='forward', cv=5)
    X_train_selected = sfs.fit_transform(X_train, y_train)
    X_test_selected = sfs.transform(X_test)

    # 打印所选的特征
    selected_features = X_train.columns[sfs.get_support()].tolist()
    print(selected_features)

    # 模型训练
    model = model_class()
    if is_svc:
        svc_model = SVC()
        model = CalibratedClassifierCV(base_estimator=svc_model)
    # model.fit(X_train_selected, y_train)

    # 自动调参
    best_model = auto_tuning(model, model_name, X_train_selected, y_train)
    # best_model.fit(X_train_selected, y_train)

    # 预测
    y_pred = best_model.predict(X_test_selected)

    # 计算准确度、精确度和召回率
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # 保存预测评估结果
    save_metrics_to_excel(model_name, accuracy, precision, recall, selected_features, output_file_path)

    # 绘制ROC曲线
    if hasattr(best_model, "predict_proba"):
        plot_and_save_roc(best_model, X_test_selected, y_test, model_name, roc_save_path)
    else:
        print(f"{model_name} does not support ROC curve plotting as it does not implement predict_proba.")

    # 输出结果
    print(f"Accuracy ({model_name}):", accuracy)
    print(f"Precision ({model_name}):", precision)
    print(f"Recall ({model_name}):", recall)

def main():
    models = [
        (LogisticRegression, 'Logistic Regression'),
        (RandomForestClassifier, 'Random Forest'),
        (SVC, 'Support Vector Machine'),
        (DecisionTreeClassifier, 'Decision Tree'),
        (KNeighborsClassifier, 'K-Nearest Neighbors'),
        (MLPClassifier, 'Neural Network'),
        (GradientBoostingClassifier, 'Gradient Boosting Tree'),
        (AdaBoostClassifier, 'AdaBoost'),
        (XGBClassifier, 'XGBoost'),
        (GaussianNB, 'Naive Bayes')
    ]

    for sheet_name in sheet_names:
        for model_class, model_name in models:
            is_svc = model_name == 'Support Vector Machine'
            train_and_evaluate_model(model_class, model_name, input_file_path, sheet_name, is_svc,)

if __name__ == '__main__':
    main()
