import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 读取Excel文件
file_path = 'D:/Analysis of diabetes/Test_8/data_normalization.xlsx'
df = pd.read_excel(file_path)

# 因变量
y = df.iloc[:, 1]

# 自变量
X = df.iloc[:, 2:28]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def forward_search(X_train, y_train, X_test, y_test):
    selected_features = []
    best_score = 0
    n_features = X.shape[1]

    while len(selected_features) < n_features:
        remaining_features = list(set(range(n_features)) - set(selected_features))
        temp_score = 0
        temp_feature = None

        for feature in remaining_features:
            candidate_features = selected_features + [feature]
            X_train_subset = X_train.iloc[:, candidate_features]
            X_test_subset = X_test.iloc[:, candidate_features]

            model = LogisticRegression()
            model.fit(X_train_subset, y_train)
            y_pred = model.predict(X_test_subset)
            score = accuracy_score(y_test, y_pred)

            if score > temp_score:
                temp_score = score
                temp_feature = feature

        if temp_score > best_score:
            best_score = temp_score
            selected_features.append(temp_feature)
        else:
            break

    return [X_train.columns[i] for i in selected_features]

best_features = forward_search(X_train, y_train, X_test, y_test)
print("最优特征:", best_features)
