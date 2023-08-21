import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from scipy.stats import randint
import datetime


def forward_search(X_train, y_train, model):
    selected_features = []
    best_score = 0
    n_features = X_train.shape[1]

    while len(selected_features) < n_features:
        remaining_features = list(set(range(n_features)) - set(selected_features))
        temp_score = 0
        temp_feature = None

        for feature in remaining_features:
            candidate_features = selected_features + [feature]
            X_train_subset = X_train.iloc[:, candidate_features]

            grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
            grid_search.fit(X_train_subset, y_train)

            score = grid_search.best_score_

            if score > temp_score:
                temp_score = score
                temp_feature = feature

        if temp_score > best_score:
            best_score = temp_score
            selected_features.append(temp_feature)
        else:
            break

    return selected_features


def lr_model(flag):
    if flag == 0:
        return LogisticRegression(penalty='l2', solver='lbfgs', max_iter=100000)
    elif flag == 1:
        return LogisticRegression(penalty='l2', solver='liblinear', max_iter=100000)
    elif flag == 2:
        return LogisticRegression(penalty='l1', solver='liblinear', max_iter=100000)
    elif flag == 3:
        return LogisticRegression(penalty='l2', solver='newton-cg', max_iter=100000)
    else:
        return LogisticRegression(penalty='l2', solver='saga', max_iter=100000)


def LR_Grid_Search(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y):
    for i in range(5):
        # Load data
        file_path = input_file_path
        sheet_name = sheet_name
        diabetes = pd.read_excel(file_path, sheet_name=sheet_name)

        # Prepare X and y
        X = diabetes[cols_x]

        y = diabetes[cols_y].values.ravel()

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)

        # Standardize data
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)

        # 创建逻辑回归模型
        lr = lr_model(i)

        # 设置超参数候选值
        param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.8, 1, 2, 4, 8, 10, 20, 16, 32, 100, 1000, 10000]}

        # 前向搜索选择最优特征
        selected_features = forward_search(X_train, y_train, lr, param_grid)

        # 使用选择的特征训练模型
        X_train_selected = X_train.iloc[:, selected_features]
        X_test_selected = X_test.iloc[:, selected_features]

        grid_search = GridSearchCV(lr, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train_selected, y_train)

        # 使用最优参数的模型进行预测
        best_lr = grid_search.best_estimator_
        y_pred = best_lr.predict(X_test_std)
        y_pred_prob = grid_search.predict_proba(X_test_std)

        # 在测试集上进行预测
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        # 输出最优的C值
        # print("Best C:", grid_search.best_params_['C'])
        print("Accuracy:", accuracy)
        # print("Precision:", precision)
        # print("Recall:", recall)

        # 计算并绘制ROC曲线
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(grid_search.classes_)):
            fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_prob[:, i], pos_label=i)
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        colors = ['blue', 'red', 'green']
        for i, color in zip(range(len(grid_search.classes_)), colors):
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
        filename = current_time.strftime("%Y%m%d%H%M%S") + ".png"
        save_path = pic_path + "LR_" + filename
        plt.savefig(save_path)
        # plt.show()
        plt.close()

        # 将Accuracy, Precision, Recall保存到指定Excel中
        ID = "liblinear_grid_" + grid_search.best_estimator_.solver
        result_df = pd.DataFrame({'ID': [ID], 'Accuracy': [accuracy], 'Precision': [precision], 'Recall': [recall],
                                  'C': [grid_search.best_params_['C']], 'save_path': [save_path]})
        excel_filename = output_file_path

        # 尝试读取已有的Excel文件
        try:
            existing_df = pd.read_excel(excel_filename)
            result_df = pd.concat([existing_df, result_df], ignore_index=True)
        except FileNotFoundError:
            print("未找到目标Excel文件")

        # 保存到目标文件
        with pd.ExcelWriter(excel_filename, mode='w', engine='openpyxl') as writer:
            result_df.to_excel(writer, sheet_name='Sheet1', index=False)
    print("逻辑回归 Grid 预测完成")


if __name__ == '__main__':
    input_file_path = "D:/Analysis of diabetes/Test_8/data_normalization.xlsx"
    sheet_name = '正常范围_中间范围'
    output_file_path = "D:/Analysis of diabetes/Test_8/预测模型结果_固测_单.xlsx"
    pic_path = "D:/Analysis of diabetes/Test_8/ROC/"
    cols_x = ['artery_caliber', 'vein_caliber', 'frac', 'AVR', 'artery_curvature', 'vein_curvature', 'artery_BSTD', 'vein_BSTD', 'artery_simple_curvatures', 'vein_simple_curvatures', 'artery_Branching_Coefficient', 'vein_Branching_Coefficient', 'artery_Num1stBa', 'vein_Num1stBa', 'artery_Branching_Angle', 'vein_Branching_Angle', 'artery_Angle_Asymmetry', 'vein_Angle_Asymmetry', 'artery_Asymmetry_Raito', 'vein_Asymmetry_Raito', 'artery_JED', 'vein_JED', 'artery_Optimal_Deviation', 'vein_Optimal_Deviation', 'vessel_length_density', 'vessel_area_density']
    cols_y = 'Result'

    LR_Grid_Search(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y)