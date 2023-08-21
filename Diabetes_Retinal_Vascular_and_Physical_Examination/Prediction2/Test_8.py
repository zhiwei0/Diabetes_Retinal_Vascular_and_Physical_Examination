import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SequentialFeatureSelector
from Analysis_and_Prediction import Prediction_Models
from scipy.stats import pearsonr

# 使用前向搜索来筛选特征并返回
# 使用sklearn组件来实现
def select_features(input_file_path, sheet_name, model_class):
    # 读取数据
    data = pd.read_excel(input_file_path, sheet_name=sheet_name)
    X = data.iloc[:, 2:]
    y = data['Result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 特征选择
    sfs = SequentialFeatureSelector(model_class(), n_features_to_select=5, direction='forward', cv=5)

    X_train_selected = sfs.fit_transform(X_train, y_train)
    X_test_selected = sfs.transform(X_test)

    # 返回所选的特征
    selected_features = X_train.columns[sfs.get_support()].tolist()
    print(selected_features)
    return selected_features

# 使用前向搜索来筛选特征并返回
# 手动实现前向选择
def perform_forward_search(input_file_path, sheet_name, model, model_name):
    from sklearn.model_selection import train_test_split, cross_val_score
    import pandas as pd

    def forward_search(X, y, model):
        remaining_features = list(X.columns)
        selected_features = []
        best_score = 0

        while remaining_features:
            scores_with_candidates = []
            for candidate in remaining_features:
                temp_features = selected_features + [candidate]
                score = cross_val_score(model(), X[temp_features], y, cv=5).mean()
                scores_with_candidates.append((score, candidate))

            scores_with_candidates.sort(reverse=True)
            best_new_score, best_candidate = scores_with_candidates[0]

            if best_new_score > best_score:
                best_score = best_new_score
                selected_features.append(best_candidate)
                remaining_features.remove(best_candidate)
            else:
                break

        return selected_features, best_score

    # 载入数据
    data = pd.read_excel(input_file_path, sheet_name=sheet_name)
    X = data.iloc[:, 2:]
    y = data['Result']

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 执行前向搜索
    best_features, best_score = forward_search(X_train, y_train, model)
    print(f"{model_name} - Best Features: {best_features}")
    print(f"{model_name} - Best Cross-Validation Score: {best_score}\n")

    return best_features

# 向Excel表格中写数据
def add_value_to_excel_column(file_path, sheet_name, columns_values):
    try:
        # 尝试读取Excel文件
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except FileNotFoundError:
        # 如果文件不存在，则创建一个空的DataFrame
        df = pd.DataFrame()

    # 在指定列的末尾添加值
    # 追加新列
    for column_name, values in columns_values.items():
        new_column = pd.Series(values, name=column_name)
        df = pd.concat([df, new_column], axis=1)

    # 保存更改到Excel文件
    df.to_excel(file_path, sheet_name=sheet_name, index=False)

# 向Excel表格中写数据
def add_value_to_excel_column1(file_path, sheet_name, new_row_data):
    # 能在同一列添加
    try:
        # 尝试读取Excel文件
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except FileNotFoundError:
        # 如果文件不存在，则创建一个空的DataFrame，其列名与new_row_data的键相匹配
        df = pd.DataFrame(columns=new_row_data.keys())

    # 追加新行
    new_row = pd.Series(new_row_data)
    df = df.append(new_row, ignore_index=True)

    # 保存更改到Excel文件
    df.to_excel(file_path, sheet_name=sheet_name, index=False)

# 使用皮尔逊相关系数筛选特征
# 对特征归一化数据不敏感
def select_features_by_pearson(input_file_path, sheet_name, n_top_features=10):
    # 载入数据
    data = pd.read_excel(input_file_path, sheet_name=sheet_name)
    X = data.iloc[:, 2:]  # 假设特征从第三列开始
    y = data['Result']   # 假设目标变量名为'Result'

    # 计算每个特征与目标变量之间的皮尔逊相关系数
    correlations = [pearsonr(X[col], y)[0] for col in X.columns]

    # 获取相关性的绝对值，并找到最高的n_top_features个特征
    top_features = [X.columns[i] for i in sorted(range(len(correlations)), key=lambda i: abs(correlations[i]), reverse=True)[:n_top_features]]

    print(f"Selected features using Pearson: {top_features}")

    return top_features

# 使用互信息筛选特征
# 与卡方检验不同，互信息不需要特征和目标变量是非负的，因此它更适合已经归一化的数据
def select_features_using_mutual_info(input_file_path, sheet_name, k=10):
    # 载入数据
    data = pd.read_excel(input_file_path, sheet_name=sheet_name)
    X = data.iloc[:, 2:]
    y = data['Result']

    # 使用互信息选择最佳特征
    # mutual_info_classif换成chi2即可使用卡方检验
    selector = SelectKBest(mutual_info_classif, k=k)
    X_new = selector.fit_transform(X, y)

    # 获取所选特征的名称
    selected_features = [column for (column, selected) in zip(X.columns, selector.get_support()) if selected]

    print(f"Selected {k} features using Mutual Information: {selected_features}")

    return selected_features



# 使用与模型相关的筛选指标的方法
def select_and_predict():
    input_file_path = "D:/Analysis of diabetes/Test_8/data_normalization.xlsx"
    sheet_names = ['正常范围_中间范围', '中间范围_糖尿病']
    output_file_path = "D:/Analysis of diabetes/Test_8/预测模型结果.xlsx"
    pic_path = "D:/Analysis of diabetes/Test_8/ROC/"

    cols_y = 'Result'

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
        columns_values = {'Sheet_Name': sheet_name}
        add_value_to_excel_column1(output_file_path,'Sheet1', columns_values)
        for model_class, model_name in models:
            # cols_x = select_features(input_file_path, sheet_name, model_class)
            cols_x = perform_forward_search(input_file_path, sheet_name, model_class, model_name)
            columns_values = {
                'model_name': model_name,
                'cols_x': cols_x
            }
            add_value_to_excel_column1(output_file_path,'Sheet1', columns_values)
            Prediction_Models.Prediction_Models(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y)


# 使用统计学相关的方法筛选指标
def select_and_predict1():
    input_file_path = "D:/Analysis of diabetes/Test_8/data_normalization.xlsx"
    sheet_names = ['正常范围_中间范围', '中间范围_糖尿病']
    output_file_path = "D:/Analysis of diabetes/Test_8/预测模型结果.xlsx"
    pic_path = "D:/Analysis of diabetes/Test_8/ROC/"

    cols_y = 'Result'

    for sheet_name in sheet_names:
        cols_x = select_features_by_pearson(input_file_path, sheet_name)
        # cols_x = select_features_using_mutual_info(input_file_path, sheet_name)

        columns_values = {
            'Sheet_Name': sheet_name,
            'cols_x': cols_x
        }
        add_value_to_excel_column1(output_file_path, 'Sheet1', columns_values)

        Prediction_Models.Prediction_Models(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y)


if __name__ == '__main__':
    # Test_8实验专用
    select_and_predict()
    # select_and_predict1()