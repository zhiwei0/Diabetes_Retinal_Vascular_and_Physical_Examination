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
sheet_name = "sheet1"

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


def save_metrics_to_excel(model_name, accuracy, precision, recall, selected_features, file_path, sheet_name):
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

    # 如果文件已经存在，则读取现有文件
    if pd.io.common.is_url(file_path) or pd.io.common.is_fsspec_url(file_path):
        raise ValueError("URLs are not supported, please provide a local file path.")

    try:
        # 使用ExcelWriter，以便在同一文件中保存多个工作表
        with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
            try:
                existing_df = pd.read_excel(file_path, sheet_name=sheet_name)
                df = pd.concat([existing_df, df], ignore_index=True)
            except FileNotFoundError:
                pass  # 工作表不存在，将创建新工作表

            # 将DataFrame保存到指定的工作表中
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    except FileNotFoundError:
        # 文件不存在，将创建新文件并保存工作表
        df.to_excel(file_path, sheet_name=sheet_name, index=False)


def logistic_regression_model(file_path):
    # 读取数据
    data = pd.read_excel(file_path, sheet_name='正常范围_中间范围')
    X = data.iloc[:, 2:]
    y = data['Result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 特征选择
    sfs = SequentialFeatureSelector(LogisticRegression(), n_features_to_select='auto', direction='forward', cv=5)
    X_train_selected = sfs.fit_transform(X_train, y_train)
    X_test_selected = sfs.transform(X_test)

    # 打印所选的特征
    selected_features = X_train.columns[sfs.get_support()].tolist()
    print(selected_features)

    # 模型训练
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train_selected, y_train)

    # 预测
    y_pred_test_lr = logistic_model.predict(X_test_selected)

    # 计算准确度、精确度和召回率
    accuracy_lr = accuracy_score(y_test, y_pred_test_lr)
    precision_lr = precision_score(y_test, y_pred_test_lr)
    recall_lr = recall_score(y_test, y_pred_test_lr)
    # 保存预测评估结果
    save_metrics_to_excel('Logistic Regression', accuracy_lr, precision_lr, recall_lr, selected_features, output_file_path, sheet_name)

    # 绘制ROC曲线
    plot_and_save_roc(logistic_model, X_test_selected, y_test, 'LR', roc_save_path)

    # 输出结果
    print("Accuracy (LR):", accuracy_lr)
    print("Precision (LR):", precision_lr)
    print("Recall (LR):", recall_lr)

def main():
    logistic_regression_model(input_file_path)

if __name__ == '__main__':
    main()