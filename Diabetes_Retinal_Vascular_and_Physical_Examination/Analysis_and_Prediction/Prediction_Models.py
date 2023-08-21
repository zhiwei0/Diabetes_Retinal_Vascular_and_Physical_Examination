from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB  # Import Gaussian Naive Bayes
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, f1_score


def select_model(i):
    if i == 0:
        return LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000000)
    elif i == 1:
        return LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000000)
    elif i == 2:
        return LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000000)
    elif i == 3:
        return LogisticRegression(penalty='l2', solver='newton-cg', max_iter=1000000)
    else:
        return LogisticRegression(penalty='l2', solver='saga', max_iter=1000000)

def LR_Grid(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y):
    for x in range(5):
        # Load data
        file_path = input_file_path
        sheet_name = sheet_name
        # sheet_name='中间范围_糖尿病'
        diabetes = pd.read_excel(file_path, sheet_name=sheet_name)

        # Prepare X and y
        X = diabetes[cols_x].values
        y = diabetes[cols_y].values.ravel()

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)

        # Standardize data
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)

        # 创建逻辑回归模型
        lr = select_model(x)

        # 设置超参数候选值
        param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.8, 1, 2, 4, 8, 10, 20, 16, 32, 50, 100, 200, 300, 400, 500, 1000, 2000, 5000, 10000, 100000]}

        # 创建网格搜索对象
        grid_search = GridSearchCV(lr, param_grid, cv=5, n_jobs=-1)

        # 在训练集上进行网格搜索
        grid_search.fit(X_train, y_train)

        # 使用最优参数的模型进行预测
        best_lr = grid_search.best_estimator_
        y_pred = best_lr.predict(X_test_std)
        y_pred_prob = grid_search.predict_proba(X_test_std)

        # 在测试集上进行预测
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # 输出最优的C值
        # print("Best C:", grid_search.best_params_['C'])
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

        # 计算并绘制ROC曲线
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(grid_search.classes_)):
            fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_prob[:, i], pos_label=i)
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        colors = ['#8A2BE2', '#7FFF00', '#FF69B4']  # 蓝紫色、查特鲁斯绿和亮粉红色
        for i, color in zip(range(len(grid_search.classes_)), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve (area = %0.2f) for class %s' % (roc_auc[i], i))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Logistic Regression')
        plt.legend(loc="lower right")

        # 将图像根据时间命名并保存到指定路径
        current_time = datetime.datetime.now()
        filename = current_time.strftime("%Y%m%d%H%M%S") + ".png"
        save_path = pic_path + "LR_" + filename
        plt.savefig(save_path)
        # plt.show()

        # 将Accuracy, Precision, Recall保存到指定Excel中
        ID = "LR_grid_" + grid_search.best_estimator_.solver
        result_df = pd.DataFrame({'ID': [ID],
                                  'Accuracy': [accuracy],
                                  'Precision': [precision],
                                  'Recall': [recall],
                                  'F1 Score': [f1],
                                  'C': [grid_search.best_params_['C']],
                                  'save_path': [save_path]
                                  })
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

        plt.close()
        print(f"LR Grid {x} is completed")

def LR_Random(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y):
    for x in range(5):
        # Load data
        file_path = input_file_path
        sheet_name = sheet_name
        diabetes = pd.read_excel(file_path, sheet_name=sheet_name)

        # Prepare X and y
        X = diabetes[cols_x].values
        y = diabetes[cols_y].values.ravel()

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

        # Standardize data
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)

        lr = select_model(x)

        # 设置超参数搜索范围
        param_distributions = {'C': uniform(loc=0, scale=1000)}

        # 创建随机搜索对象
        random_search = RandomizedSearchCV(lr, param_distributions, cv=5, n_jobs=-1, n_iter=50)

        # 在训练集上进行随机搜索
        random_search.fit(X_train_std, y_train)

        # 使用最优参数的模型进行预测
        best_lr = random_search.best_estimator_
        y_pred = best_lr.predict(X_test_std)
        y_pred_prob = random_search.predict_proba(X_test_std)

        # 在测试集上进行预测
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # 输出最优的C值
        # print("Best C:", grid_search.best_params_['C'])
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

        # 计算并绘制ROC曲线
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(random_search.classes_)):
            fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_prob[:, i], pos_label=i)
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        colors = ['#8A2BE2', '#7FFF00', '#FF69B4']  # 蓝紫色、查特鲁斯绿和亮粉红色
        for i, color in zip(range(len(random_search.classes_)), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve (area = %0.2f) for class %s' % (roc_auc[i], i))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Logistic Regression')
        plt.legend(loc="lower right")

        # 将图像根据时间命名并保存到指定路径
        current_time = datetime.datetime.now()
        filename = current_time.strftime("%Y%m%d%H%M%S") + ".png"
        save_path = pic_path + "LR_" + filename
        plt.savefig(save_path)
        # plt.show()

        # 将Accuracy, Precision, Recall保存到指定Excel中
        ID = "LR_random_" + random_search.best_estimator_.solver
        result_df = pd.DataFrame({'ID': [ID],
                                  'Accuracy': [accuracy],
                                  'Precision': [precision],
                                  'Recall': [recall],
                                  'F1 Score': [f1],
                                  'C': [random_search.best_params_['C']],
                                  'save_path': [save_path]
                                  })
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

        plt.close()
        print(f"LR Random {x} is completed")


def RF_Grid(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y):
    # Load data
    file_path = input_file_path
    sheet_name = sheet_name
    diabetes = pd.read_excel(file_path, sheet_name=sheet_name)

    # Prepare X and y
    X = diabetes[cols_x].values

    y = diabetes[cols_y].values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
    # rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=1)
    # rf.fit(X_train_std, y_train)
    rf = RandomForestClassifier(random_state=1)

    # 定义参数网格
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    # 使用网格搜索进行参数选择
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
    grid_search.fit(X_train_std, y_train)

    y_pred = grid_search.predict(X_test_std)
    y_pred_prob = grid_search.predict_proba(X_test_std)

    # 在测试集上进行预测
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # 输出最优的C值
    # print("Best C:", grid_search.best_params_['C'])
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # 计算并绘制ROC曲线
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(grid_search.classes_)):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_prob[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    colors = ['#8A2BE2', '#7FFF00', '#FF69B4']  # 蓝紫色、查特鲁斯绿和亮粉红色
    for i, color in zip(range(len(grid_search.classes_)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve (area = %0.2f) for class %s' % (roc_auc[i], i))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Random Forest')
    plt.legend(loc="lower right")

    # 将图像根据时间命名并保存到指定路径
    current_time = datetime.datetime.now()
    filename = current_time.strftime("%Y%m%d%H%M%S") + ".png"
    save_path = pic_path + "random_forest_grid_" + filename
    plt.savefig(save_path)

    # plt.show()

    # 将Accuracy, Precision, Recall保存到指定Excel中
    ID = "Random_Rorest_grid_"
    result_df = pd.DataFrame({'ID': [ID],
                              'Accuracy': [accuracy],
                              'Precision': [precision],
                              'Recall': [recall],
                              'F1 Score': [f1],
                              'C': [grid_search.best_estimator_],
                              'save_path': [save_path]
                              })
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

    plt.close()
    print(f"RF Grid is completed")

def RF_Random(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y):
    # Load data
    file_path = input_file_path
    sheet_name = sheet_name
    diabetes = pd.read_excel(file_path, sheet_name=sheet_name)

    # Prepare X and y
    X = diabetes[cols_x].values
    y = diabetes[cols_y].values.ravel()

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
        'n_estimators': randint(1, 1000),  # 决策树的数量范围在10到200之间随机选择
        'max_depth': randint(1, 1000),  # 最大深度范围在1到20之间随机选择
        'min_samples_split': randint(2, 1000),  # 内部节点分裂所需的最小样本数范围在2到20之间随机选择
        'min_samples_leaf': randint(1, 1000),  # 叶节点所需的最小样本数范围在1到20之间随机选择
        'max_features': ['sqrt', 'log2'],  # 最佳分割特征数从三个选项中随机选择
        'bootstrap': [True, False],  # 是否使用有放回抽样随机选择
        'class_weight': [None, 'balanced']  # 类别权重从两个选项中随机选择
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
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # 输出最优的C值
    # print("Best C:", grid_search.best_params_['C'])
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # 计算并绘制ROC曲线
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(random_search.classes_)):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_prob[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    colors = ['#8A2BE2', '#7FFF00', '#FF69B4']  # 蓝紫色、查特鲁斯绿和亮粉红色
    for i, color in zip(range(len(random_search.classes_)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve (area = %0.2f) for class %s' % (roc_auc[i], i))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Random Forest')
    plt.legend(loc="lower right")

    # 将图像根据时间命名并保存到指定路径
    current_time = datetime.datetime.now()
    filename = current_time.strftime("%Y%m%d%H%M%S") + ".png"
    save_path = pic_path + "random_forest_" + filename
    plt.savefig(save_path)

    # plt.show()

    # 将Accuracy, Precision, Recall保存到指定Excel中
    ID = "Random_Forest_random_"
    result_df = pd.DataFrame({'ID': [ID],
                              'Accuracy': [accuracy],
                              'Precision': [precision],
                              'Recall': [recall],
                              'F1 Score': [f1],
                              'C': [random_search.best_estimator_],
                              'save_path': [save_path]
                              })
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

    plt.close()
    print(f"RF Random is completed")


def SVM_Grid(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y):
    # 判断cols_x中指标个数，若只有一个则不执行
    # if len(cols_x)<=2 :
    #     print("cols_x 个数必须大于2")
    #     return
    # Load data
    file_path = input_file_path
    sheet_name = sheet_name

    diabetes = pd.read_excel(file_path, sheet_name=sheet_name)

    # Prepare X and y
    X = diabetes[cols_x].values

    y = diabetes[cols_y].values.ravel()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Scale the features
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # # Create an 3_SVM classifier
    svm = SVC(random_state=1, probability=True)
    # param_grid = {
    #     'C': [0.5, 1.0, 2.0, 3.0],
    #     'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    #     'degree': [3, 6, 9],
    #     'coef0': [0.0, 1.0, 2.0]
    # }
    # 优化后的网格搜索参数设置
    param_grid = [
        {'kernel': ['linear'], 'C': [0.5, 1.0, 2.0, 3.0]},
        {'kernel': ['rbf'], 'C': [0.5, 1.0, 2.0, 3.0]},
        {'kernel': ['poly'], 'C': [0.5, 1.0, 2.0, 3.0], 'degree': [3, 6, 9], 'coef0': [0.0, 1.0, 2.0]},
        {'kernel': ['sigmoid'], 'C': [0.5, 1.0, 2.0, 3.0], 'coef0': [0.0, 1.0, 2.0]}
    ]

    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5)
    grid_search.fit(X_train_std, y_train)
    y_pred = grid_search.predict(X_test_std)
    y_pred_prob = grid_search.predict_proba(X_test_std)

    # 获取最优模型
    best_svm_classifier = grid_search.best_estimator_

    # 使用最优模型在测试集上进行拟合和预测
    best_svm_classifier.fit(X_train_std, y_train)
    y_pred_test = best_svm_classifier.predict(X_test_std)
    y_pred_prob_test = best_svm_classifier.predict_proba(X_test_std)

    # Calculate ROC curve and AUC for test set
    fpr_test = dict()
    tpr_test = dict()
    roc_auc_test = dict()

    for i in range(best_svm_classifier.classes_.size):
        fpr_test[i], tpr_test[i], _ = roc_curve(y_test, y_pred_prob_test[:, i], pos_label=i)
        roc_auc_test[i] = auc(fpr_test[i], tpr_test[i])

        # Calculate accuracy, recall, and precision for test set
    accuracy_test = accuracy_score(y_test, y_pred_test)
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    precision_test = precision_score(y_test, y_pred_test, average='weighted')
    f1 = f1_score(y_test, y_pred_test, average='weighted')

    # 输出最优的C值
    # print("Best Parameters:", grid_search.best_params_)
    print("Accuracy:", accuracy_test)
    print("Precision:", precision_test)
    print("Recall:", recall_test)
    print("F1 Score:", f1)

    # Plotting the ROC curves for test set
    plt.figure()
    colors = ['#8A2BE2', '#7FFF00', '#FF69B4']  # 蓝紫色、查特鲁斯绿和亮粉红色
    for i, color in zip(range(best_svm_classifier.classes_.size), colors):
        plt.plot(fpr_test[i], tpr_test[i], color=color, lw=2,
                 label='ROC curve (area = %0.2f) for class %s' % (roc_auc_test[i], i))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Support Vector Machine')
    plt.legend(loc="lower right")

    # 将图像根据时间命名并保存到指定路径
    current_time = datetime.datetime.now()
    filename = current_time.strftime("%Y%m%d%H%M%S") + ".png"
    save_path = pic_path + "SVM_Grid_Search_" + filename
    plt.savefig(save_path)

    # plt.show()

    # 将Accuracy, Precision, Recall保存到指定Excel中
    ID = "SVM_grid_"
    result_df = pd.DataFrame({
         'ID': [ID],
         'Accuracy': [accuracy_test],
         'Precision': [precision_test],
         'Recall': [recall_test],
         'F1 Score': [f1],
         'C': [grid_search.best_estimator_],
         'save_path': [save_path]
         })
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

    plt.close()
    print(f"SVM Grid is completed")


def SVM_Random(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y):
    # 判断cols_x中指标个数，若只有一个则不执行
    # if len(cols_x) <= 2:
    #     print("cols_x 个数必须大于2")
    #     return

    # Load data
    file_path = input_file_path
    sheet_name = sheet_name
    diabetes = pd.read_excel(file_path, sheet_name=sheet_name)

    # Prepare X and y
    X = diabetes[cols_x].values
    y = diabetes[cols_y].values.ravel()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Scale the features
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # # Create an 3_SVM classifier
    param_dist = {
        'C': randint(1, 11),  # 在[1, 10]范围内随机选择整数作为C的值
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'degree': randint(1, 10),  # 在[3, 9]范围内随机选择整数作为degree的值
        'coef0': randint(0, 5)  # 在[0, 2]范围内随机选择整数作为coef0的值
    }

    # 创建随机搜索对象
    svm = SVC(random_state=1, probability=True)
    # n_iter参数指定了随机搜索要执行的随机采样次数，可以根据需要设置为合适的值
    random_search = RandomizedSearchCV(svm, param_distributions=param_dist, n_iter=20, cv=5, random_state=1)

    # 在训练数据上进行随机搜索调参
    random_search.fit(X_train_std, y_train)

    best_svm_classifier = random_search.best_estimator_
    best_svm_classifier.fit(X_train_std, y_train)

    # 进行预测
    y_pred_test = best_svm_classifier.predict(X_test_std)
    y_pred_prob_test = best_svm_classifier.predict_proba(X_test_std)

    # 计算ROC曲线和AUC
    fpr_test = dict()
    tpr_test = dict()
    roc_auc_test = dict()

    for i in range(best_svm_classifier.classes_.size):
        fpr_test[i], tpr_test[i], _ = roc_curve(y_test, y_pred_prob_test[:, i], pos_label=i)
        roc_auc_test[i] = auc(fpr_test[i], tpr_test[i])
        # 计算性能指标
    accuracy_test = accuracy_score(y_test, y_pred_test)
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    precision_test = precision_score(y_test, y_pred_test, average='weighted')
    f1 = f1_score(y_test, y_pred_test, average='weighted')

    # 输出最优的C值
    # print("Best Parameters:", grid_search.best_params_)
    print("Accuracy:", accuracy_test)
    print("Precision:", precision_test)
    print("Recall:", recall_test)
    print("F1 Score:", f1)

    # Plotting the ROC curves for test set
    plt.figure()
    colors = ['#8A2BE2', '#7FFF00', '#FF69B4']  # 蓝紫色、查特鲁斯绿和亮粉红色
    for i, color in zip(range(best_svm_classifier.classes_.size), colors):
        plt.plot(fpr_test[i], tpr_test[i], color=color, lw=2,
                 label='ROC curve (area = %0.2f) for class %s' % (roc_auc_test[i], i))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Support Vector Machine')
    plt.legend(loc="lower right")

    # 将图像根据时间命名并保存到指定路径
    current_time = datetime.datetime.now()
    filename = current_time.strftime("%Y%m%d%H%M%S") + ".png"
    save_path = pic_path + "SVM_Random_Search_" + filename
    plt.savefig(save_path)

    # plt.show()

    # 将Accuracy, Precision, Recall保存到指定Excel中
    ID = "SVM_random_"
    result_df = pd.DataFrame({'ID': [ID],
                              'Accuracy': [accuracy_test],
                              'Precision': [precision_test],
                              'Recall': [recall_test],
                              'F1 Score': [f1],
                              'C': [random_search.best_estimator_],
                              'save_path': [save_path]
                              })
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

    plt.close()
    print(f"SVM Random is complete")


def DT_Grid(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y):
    # Load data
    file_path = input_file_path
    sheet_name = sheet_name

    diabetes = pd.read_excel(file_path, sheet_name=sheet_name)

    # Prepare X and y
    X = diabetes[cols_x].values
    y = diabetes[cols_y].values.ravel()

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
    f1 = f1_score(y_test, y_pred_test, average='weighted')

    fpr_test = dict()
    tpr_test = dict()
    roc_auc_test = dict()

    for i in range(best_dt_classifier.classes_.size):
        fpr_test[i], tpr_test[i], _ = roc_curve(y_test, y_pred_prob_test[:, i], pos_label=i)
        roc_auc_test[i] = auc(fpr_test[i], tpr_test[i])

    # 输出最优的C值
    # print("Best C:", grid_search.best_params_['C'])
    print("Accuracy:", accuracy_test)
    print("Precision:", precision_test)
    print("Recall:", recall_test)
    print("F1 Score:", f1)

    plt.figure()
    colors = ['#8A2BE2', '#7FFF00', '#FF69B4']  # 蓝紫色、查特鲁斯绿和亮粉红色
    for i, color in zip(range(best_dt_classifier.classes_.size), colors):
        plt.plot(fpr_test[i], tpr_test[i], color=color, lw=2,
                 label='ROC curve (area = %0.2f) for class %s' % (roc_auc_test[i], i))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Decision Tree')
    plt.legend(loc="lower right")

    current_time = datetime.datetime.now()
    filename = current_time.strftime("%Y%m%d%H%M%S") + ".png"
    save_path = pic_path + "Decision_Tree_Grid_Search_" + filename
    plt.savefig(save_path)

    # plt.show()

    # 将Accuracy, Precision, Recall保存到指定Excel中
    ID = "Decision_Tree_grid_"
    result_df = pd.DataFrame({'ID': [ID],
                              'Accuracy': [accuracy_test],
                              'Precision': [precision_test],
                              'Recall': [recall_test],
                              'F1 Score': [f1],
                              'C': [grid_search.best_estimator_],
                              'save_path': [save_path]
                              })
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

    plt.close()
    print(f"DT Grid is complete")


def DT_Random(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y):
    # Load data
    file_path = input_file_path
    sheet_name = sheet_name
    diabetes = pd.read_excel(file_path, sheet_name=sheet_name)

    # Prepare X and y
    X = diabetes[cols_x].values
    y = diabetes[cols_y].values.ravel()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Scale the features
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # 创建决策树模型对象：
    dt = DecisionTreeClassifier(random_state=1)

    # 定义决策树的参数空间（可选，可以根据需要进行调参）：
    param_dist = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': randint(1, 10),
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 11)
    }

    # 创建随机搜索对象：
    random_search = RandomizedSearchCV(dt, param_distributions=param_dist, n_iter=20, cv=5, random_state=1)

    # 进行随机搜索调参：
    random_search.fit(X_train_std, y_train)

    # 获取最优模型：
    best_dt_classifier = random_search.best_estimator_

    # 进行预测：
    y_pred_test = best_dt_classifier.predict(X_test_std)
    y_pred_prob_test = best_dt_classifier.predict_proba(X_test_std)

    # 计算性能指标和ROC曲线：

    accuracy_test = accuracy_score(y_test, y_pred_test)
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    precision_test = precision_score(y_test, y_pred_test, average='weighted')
    f1 = f1_score(y_test, y_pred_test, average='weighted')

    # 打印结果
    # print("Best C:", grid_search.best_params_['C'])
    print("Accuracy:", accuracy_test)
    print("Precision:", precision_test)
    print("Recall:", recall_test)
    print("F1 Score:", f1)

    fpr_test = dict()
    tpr_test = dict()
    roc_auc_test = dict()

    for i in range(best_dt_classifier.classes_.size):
        fpr_test[i], tpr_test[i], _ = roc_curve(y_test, y_pred_prob_test[:, i], pos_label=i)
        roc_auc_test[i] = auc(fpr_test[i], tpr_test[i])

    plt.figure()
    colors = ['#8A2BE2', '#7FFF00', '#FF69B4']  # 蓝紫色、查特鲁斯绿和亮粉红色
    for i, color in zip(range(best_dt_classifier.classes_.size), colors):
        plt.plot(fpr_test[i], tpr_test[i], color=color, lw=2,
                 label='ROC curve (area = %0.2f) for class %s' % (roc_auc_test[i], i))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Decision Tree')
    plt.legend(loc="lower right")

    current_time = datetime.datetime.now()
    filename = current_time.strftime("%Y%m%d%H%M%S") + ".png"
    save_path = pic_path + "Decision_Tree_Random_Search_" + filename
    plt.savefig(save_path)

    # plt.show()

    # 将Accuracy, Precision, Recall保存到指定Excel中
    ID = "Decision_Tree_random_"
    result_df = pd.DataFrame({'ID': [ID],
                              'Accuracy': [accuracy_test],
                              'Precision': [precision_test],
                              'Recall': [recall_test],
                              'F1 Score': [f1],
                              'C': [random_search.best_estimator_],
                              'save_path': [save_path]
                              })
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

    plt.close()
    print(f"DT Random is complete")


def KNN_Grid(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y):
    # Load data
    file_path = input_file_path
    sheet_name = sheet_name
    diabetes = pd.read_excel(file_path, sheet_name=sheet_name)

    # Prepare X and y
    X = diabetes[cols_x].values
    y = diabetes[cols_y].values.ravel()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Scale the features
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # Create a K-Nearest Neighbors (5_KNN) classifier
    knn = KNeighborsClassifier()
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # 1: Manhattan distance, 2: Euclidean distance
    }

    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5)
    grid_search.fit(X_train_std, y_train)
    y_pred = grid_search.predict(X_test_std)
    y_pred_prob = grid_search.predict_proba(X_test_std)

    # Get the best 5_KNN model
    best_knn_classifier = grid_search.best_estimator_

    # Use the best model to predict on the test set
    y_pred_test = best_knn_classifier.predict(X_test_std)
    y_pred_prob_test = best_knn_classifier.predict_proba(X_test_std)

    # Calculate accuracy, recall, and precision for the test set
    accuracy_test = accuracy_score(y_test, y_pred_test)
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    precision_test = precision_score(y_test, y_pred_test, average='weighted')
    f1 = f1_score(y_test, y_pred_test, average='weighted')

    # 打印结果
    # print("Best C:", grid_search.best_params_['C'])
    print("Accuracy:", accuracy_test)
    print("Precision:", precision_test)
    print("Recall:", recall_test)
    print("F1 Score:", f1)

    # Calculate ROC curve and AUC for the test set
    fpr_test = dict()
    tpr_test = dict()
    roc_auc_test = dict()

    for i in range(best_knn_classifier.classes_.size):
        fpr_test[i], tpr_test[i], _ = roc_curve(y_test, y_pred_prob_test[:, i], pos_label=i)
        roc_auc_test[i] = auc(fpr_test[i], tpr_test[i])

    # Plot the ROC curves for the test set
    plt.figure()
    colors = ['#8A2BE2', '#7FFF00', '#FF69B4']  # 蓝紫色、查特鲁斯绿和亮粉红色
    for i, color in zip(range(best_knn_classifier.classes_.size), colors):
        plt.plot(fpr_test[i], tpr_test[i], color=color, lw=2,
                 label='ROC curve (area = %0.2f) for class %s' % (roc_auc_test[i], i))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for K-Nearest Neighbors')
    plt.legend(loc="lower right")

    # Save the plot with a filename based on the current time
    current_time = datetime.datetime.now()
    filename = current_time.strftime("%Y%m%d%H%M%S") + ".png"
    save_path = pic_path + "KNN_Grid_Search_" + filename
    plt.savefig(save_path)

    # plt.show()

    # 将Accuracy, Precision, Recall保存到指定Excel中
    ID = "KNN_grid_"
    result_df = pd.DataFrame({'ID': [ID],
                              'Accuracy': [accuracy_test],
                              'Precision': [precision_test],
                              'Recall': [recall_test],
                              'F1 Score': [f1],
                              'C': [grid_search.best_estimator_],
                              'save_path': [save_path]
                              })
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

    plt.close()
    print(f"KNN Grid is complete")


def KNN_Random(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y):
    # Load data
    file_path = input_file_path
    sheet_name = sheet_name

    diabetes = pd.read_excel(file_path, sheet_name=sheet_name)

    # Prepare X and y
    X = diabetes[cols_x].values
    y = diabetes[cols_y].values.ravel()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Scale the features
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # Create a 5_KNN classifier
    knn = KNeighborsClassifier()

    # Define the parameter grid for RandomizedSearchCV
    param_dist = {
        'n_neighbors': np.arange(1, 11),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2]
    }

    # Create RandomizedSearchCV object
    random_search = RandomizedSearchCV(knn, param_distributions=param_dist, n_iter=50, cv=5, random_state=1)

    # Perform Randomized Search for hyperparameter tuning
    random_search.fit(X_train_std, y_train)

    # Get the best model
    best_knn_classifier = random_search.best_estimator_

    # Make predictions on the test set
    y_pred_test = best_knn_classifier.predict(X_test_std)
    y_pred_prob_test = best_knn_classifier.predict_proba(X_test_std)

    # Calculate accuracy, recall, and precision for test set
    accuracy_test = accuracy_score(y_test, y_pred_test)
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    precision_test = precision_score(y_test, y_pred_test, average='weighted')
    f1 = f1_score(y_test, y_pred_test, average='weighted')

    # 打印结果
    # print("Best C:", grid_search.best_params_['C'])
    print("Accuracy:", accuracy_test)
    print("Precision:", precision_test)
    print("Recall:", recall_test)
    print("F1 Score:", f1)

    # Calculate ROC curve and AUC for test set
    fpr_test = dict()
    tpr_test = dict()
    roc_auc_test = dict()

    for i in range(best_knn_classifier.classes_.size):
        fpr_test[i], tpr_test[i], _ = roc_curve(y_test, y_pred_prob_test[:, i], pos_label=i)
        roc_auc_test[i] = auc(fpr_test[i], tpr_test[i])

    # Plotting the ROC curves for test set
    plt.figure()
    colors = ['#8A2BE2', '#7FFF00', '#FF69B4']  # 蓝紫色、查特鲁斯绿和亮粉红色
    for i, color in zip(range(best_knn_classifier.classes_.size), colors):
        plt.plot(fpr_test[i], tpr_test[i], color=color, lw=2,
                 label='ROC curve (area = %0.2f) for class %s' % (roc_auc_test[i], i))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for K-Nearest Neighbors')
    plt.legend(loc="lower right")

    # Save the plot with a filename based on the current time
    current_time = datetime.datetime.now()
    filename = current_time.strftime("%Y%m%d%H%M%S") + ".png"
    save_path = pic_path + "KNN_Random_Search_" + filename
    plt.savefig(save_path)

    # plt.show()

    # 将Accuracy, Precision, Recall保存到指定Excel中
    ID = "KNN_random_"
    result_df = pd.DataFrame({'ID': [ID],
                              'Accuracy': [accuracy_test],
                              'Precision': [precision_test],
                              'Recall': [recall_test],
                              'F1 Score': [f1],
                              'C': [random_search.best_estimator_],
                              'save_path': [save_path]
                              })
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

    plt.close()
    print(f"KNN Random is complete")


def NN_Grid(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y):
    # Load data
    file_path = input_file_path
    sheet_name = sheet_name
    diabetes = pd.read_excel(file_path, sheet_name=sheet_name)

    # Prepare X and y
    X = diabetes[cols_x].values
    y = diabetes[cols_y].values.ravel()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Scale the features
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # Create a Multi-Layer Perceptron (MLP) classifier
    mlp = MLPClassifier(random_state=1, max_iter=100000)

    param_grid = {
        'hidden_layer_sizes': [(100,), (50, 50), (50, 25, 10)],  # Number of neurons in each hidden layer
        'activation': ['relu', 'tanh', 'logistic'],  # Activation function
        'alpha': [0.0001, 0.001, 0.01],  # L2 regularization parameter
    }
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5)
    grid_search.fit(X_train_std, y_train)
    y_pred = grid_search.predict(X_test_std)
    y_pred_prob = grid_search.predict_proba(X_test_std)

    # Get the best MLP model
    best_mlp_classifier = grid_search.best_estimator_

    # Use the best model to predict on the test set
    y_pred_test = best_mlp_classifier.predict(X_test_std)
    y_pred_prob_test = best_mlp_classifier.predict_proba(X_test_std)

    # Calculate accuracy, recall, and precision for the test set
    accuracy_test = accuracy_score(y_test, y_pred_test)
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    precision_test = precision_score(y_test, y_pred_test, average='weighted')
    f1 = f1_score(y_test, y_pred_test, average='weighted')

    # 打印结果
    # print("Best C:", grid_search.best_params_['C'])
    print("Accuracy:", accuracy_test)
    print("Precision:", precision_test)
    print("Recall:", recall_test)
    print("F1 Score:", f1)

    # Calculate ROC curve and AUC for the test set
    fpr_test = dict()
    tpr_test = dict()
    roc_auc_test = dict()

    for i in range(best_mlp_classifier.classes_.size):
        fpr_test[i], tpr_test[i], _ = roc_curve(y_test, y_pred_prob_test[:, i], pos_label=i)
        roc_auc_test[i] = auc(fpr_test[i], tpr_test[i])

    # Plot the ROC curves for the test set
    plt.figure()
    colors = ['#8A2BE2', '#7FFF00', '#FF69B4']  # 蓝紫色、查特鲁斯绿和亮粉红色
    for i, color in zip(range(best_mlp_classifier.classes_.size), colors):
        plt.plot(fpr_test[i], tpr_test[i], color=color, lw=2,
                 label='ROC curve (area = %0.2f) for class %s' % (roc_auc_test[i], i))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Neural Network')
    plt.legend(loc="lower right")

    # Save the plot with a filename based on the current time
    current_time = datetime.datetime.now()
    filename = current_time.strftime("%Y%m%d%H%M%S") + ".png"
    save_path = pic_path + "MLP_Grid_Search_" + filename
    plt.savefig(save_path)

    # plt.show()

    # 将Accuracy, Precision, Recall保存到指定Excel中
    ID = "Neural_Network_grid_"
    result_df = pd.DataFrame({'ID': [ID],
                              'Accuracy': [accuracy_test],
                              'Precision': [precision_test],
                              'Recall': [recall_test],
                              'F1 Score': [f1],
                              'C': [grid_search.best_estimator_],
                              'save_path': [save_path]
                              })
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

    plt.close()
    print(f"NN Grid is complete")


def NN_Random(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y):
    # Load data
    file_path = input_file_path
    sheet_name = sheet_name
    diabetes = pd.read_excel(file_path, sheet_name=sheet_name)

    # Prepare X and y
    X = diabetes[cols_x].values
    y = diabetes[cols_y].values.ravel()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Scale the features
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # Create a Multi-Layer Perceptron (MLP) classifier
    mlp = MLPClassifier(random_state=1, max_iter=100000)

    param_dist = {
        'hidden_layer_sizes': [(100,), (50, 50), (50, 25, 10)],  # Number of neurons in each hidden layer
        'activation': ['relu', 'tanh', 'logistic'],  # Activation function
        'alpha': np.logspace(-5, 1, num=7)  # L2 regularization parameter (log scale)
    }

    random_search = RandomizedSearchCV(estimator=mlp, param_distributions=param_dist, n_iter=50, cv=5)
    random_search.fit(X_train_std, y_train)
    y_pred = random_search.predict(X_test_std)
    y_pred_prob = random_search.predict_proba(X_test_std)

    # Get the best MLP model
    best_mlp_classifier = random_search.best_estimator_

    # Use the best model to predict on the test set
    y_pred_test = best_mlp_classifier.predict(X_test_std)
    y_pred_prob_test = best_mlp_classifier.predict_proba(X_test_std)

    # Calculate accuracy, recall, and precision for the test set
    accuracy_test = accuracy_score(y_test, y_pred_test)
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    precision_test = precision_score(y_test, y_pred_test, average='weighted')
    f1 = f1_score(y_test, y_pred_test, average='weighted')

    # 打印结果
    # print("Best C:", grid_search.best_params_['C'])
    print("Accuracy:", accuracy_test)
    print("Precision:", precision_test)
    print("Recall:", recall_test)
    print("F1 Score:", f1)

    # Calculate ROC curve and AUC for the test set
    fpr_test = dict()
    tpr_test = dict()
    roc_auc_test = dict()

    for i in range(best_mlp_classifier.classes_.size):
        fpr_test[i], tpr_test[i], _ = roc_curve(y_test, y_pred_prob_test[:, i], pos_label=i)
        roc_auc_test[i] = auc(fpr_test[i], tpr_test[i])

    # Plot the ROC curves for the test set
    plt.figure()
    colors = ['#8A2BE2', '#7FFF00', '#FF69B4']  # 蓝紫色、查特鲁斯绿和亮粉红色
    for i, color in zip(range(best_mlp_classifier.classes_.size), colors):
        plt.plot(fpr_test[i], tpr_test[i], color=color, lw=2,
                 label='ROC curve (area = %0.2f) for class %s' % (roc_auc_test[i], i))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Neural Network')
    plt.legend(loc="lower right")

    # Save the plot with a filename based on the current time
    current_time = datetime.datetime.now()
    filename = current_time.strftime("%Y%m%d%H%M%S") + ".png"
    save_path = pic_path + "MLP_Randomized_Search_" + filename
    plt.savefig(save_path)

    # plt.show()

    # 将Accuracy, Precision, Recall保存到指定Excel中
    ID = "Neural_Network_random_"
    result_df = pd.DataFrame({'ID': [ID],
                              'Accuracy': [accuracy_test],
                              'Precision': [precision_test],
                              'Recall': [recall_test],
                              'F1 Score': [f1],
                              'C': [random_search.best_estimator_],
                              'save_path': [save_path]
                              })
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

    plt.close()
    print("NN Random is complete")


def GBT_Grid(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y):
    # Load data
    file_path = input_file_path
    sheet_name = sheet_name
    diabetes = pd.read_excel(file_path, sheet_name=sheet_name)

    # Prepare X and y
    X = diabetes[cols_x].values
    y = diabetes[cols_y].values.ravel()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Scale the features
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # Create a Gradient Boosting Tree (8_GBT) classifier
    gbt = GradientBoostingClassifier(random_state=1)

    param_grid = {
        'n_estimators': [50, 100, 200],  # Number of boosting stages to be run
        'learning_rate': [0.01, 0.1, 0.2],  # Step size at each iteration
        'max_depth': [3, 5, 7],  # Maximum depth of the individual trees
    }

    grid_search = GridSearchCV(estimator=gbt, param_grid=param_grid, cv=5)
    grid_search.fit(X_train_std, y_train)
    y_pred = grid_search.predict(X_test_std)
    y_pred_prob = grid_search.predict_proba(X_test_std)

    # Get the best 8_GBT model
    best_gbt_classifier = grid_search.best_estimator_

    # Use the best model to predict on the test set
    y_pred_test = best_gbt_classifier.predict(X_test_std)
    y_pred_prob_test = best_gbt_classifier.predict_proba(X_test_std)

    # Calculate accuracy, recall, and precision for the test set
    accuracy_test = accuracy_score(y_test, y_pred_test)
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    precision_test = precision_score(y_test, y_pred_test, average='weighted')
    f1 = f1_score(y_test, y_pred_test, average='weighted')

    # 打印结果
    # print("Best C:", grid_search.best_params_['C'])
    print("Accuracy:", accuracy_test)
    print("Precision:", precision_test)
    print("Recall:", recall_test)
    print("F1 Score:", f1)

    # Calculate ROC curve and AUC for the test set
    fpr_test = dict()
    tpr_test = dict()
    roc_auc_test = dict()

    for i in range(best_gbt_classifier.classes_.size):
        fpr_test[i], tpr_test[i], _ = roc_curve(y_test, y_pred_prob_test[:, i], pos_label=i)
        roc_auc_test[i] = auc(fpr_test[i], tpr_test[i])

    # Plotting the ROC curves for the test set
    plt.figure()
    colors = ['#8A2BE2', '#7FFF00', '#FF69B4']  # 蓝紫色、查特鲁斯绿和亮粉红色
    for i, color in zip(range(best_gbt_classifier.classes_.size), colors):
        plt.plot(fpr_test[i], tpr_test[i], color=color, lw=2,
                 label='ROC curve (area = %0.2f) for class %s' % (roc_auc_test[i], i))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Gradient Boosting Tree')
    plt.legend(loc="lower right")

    # Save the plot with a filename based on the current time
    current_time = datetime.datetime.now()
    filename = current_time.strftime("%Y%m%d%H%M%S") + ".png"
    save_path = pic_path + "GBT_Grid_Search_" + filename
    plt.savefig(save_path)

    # plt.show()

    # 将Accuracy, Precision, Recall保存到指定Excel中
    ID = "GBT_grid_"
    result_df = pd.DataFrame({'ID': [ID],
                              'Accuracy': [accuracy_test],
                              'Precision': [precision_test],
                              'Recall': [recall_test],
                              'F1 Score': [f1],
                              'C': [grid_search.best_estimator_],
                              'save_path': [save_path]
                              })
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

    plt.close()
    print("GBT Grid is complete")


def GBT_Random(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y):
    # Load data
    file_path = input_file_path
    sheet_name = sheet_name
    diabetes = pd.read_excel(file_path, sheet_name=sheet_name)

    # Prepare X and y
    X = diabetes[cols_x].values
    y = diabetes[cols_y].values.ravel()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Scale the features
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # Create a Gradient Boosting Tree (8_GBT) classifier
    gbt = GradientBoostingClassifier(random_state=1)

    param_dist = {
        'n_estimators': randint(50, 201),  # 在[50, 200]范围内随机选择整数作为n_estimators的值
        'learning_rate': [0.01, 0.1, 0.2],  # Step size at each iteration
        'max_depth': randint(3, 8),  # 在[3, 7]范围内随机选择整数作为max_depth的值
    }

    # n_iter参数指定了随机搜索要执行的随机采样次数，可以根据需要设置为合适的值
    random_search = RandomizedSearchCV(gbt, param_distributions=param_dist, n_iter=20, cv=5, random_state=1)

    random_search.fit(X_train_std, y_train)

    best_gbt_classifier = random_search.best_estimator_
    best_gbt_classifier.fit(X_train_std, y_train)

    # 进行预测
    y_pred_test = best_gbt_classifier.predict(X_test_std)
    y_pred_prob_test = best_gbt_classifier.predict_proba(X_test_std)

    # 计算性能指标
    accuracy_test = accuracy_score(y_test, y_pred_test)
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    precision_test = precision_score(y_test, y_pred_test, average='weighted')
    f1 = f1_score(y_test, y_pred_test, average='weighted')

    # 打印结果
    # print("Best C:", grid_search.best_params_['C'])
    print("Accuracy:", accuracy_test)
    print("Precision:", precision_test)
    print("Recall:", recall_test)
    print("F1 Score:", f1)

    # 计算ROC曲线和AUC
    fpr_test = dict()
    tpr_test = dict()
    roc_auc_test = dict()

    for i in range(best_gbt_classifier.classes_.size):
        fpr_test[i], tpr_test[i], _ = roc_curve(y_test, y_pred_prob_test[:, i], pos_label=i)
        roc_auc_test[i] = auc(fpr_test[i], tpr_test[i])

    # Plotting the ROC curves for test set
    plt.figure()
    colors = ['#8A2BE2', '#7FFF00', '#FF69B4']  # 蓝紫色、查特鲁斯绿和亮粉红色
    for i, color in zip(range(best_gbt_classifier.classes_.size), colors):
        plt.plot(fpr_test[i], tpr_test[i], color=color, lw=2,
                 label='ROC curve (area = %0.2f) for class %s' % (roc_auc_test[i], i))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Gradient Boosting Tree')
    plt.legend(loc="lower right")

    # 将图像根据时间命名并保存到指定路径
    current_time = datetime.datetime.now()
    filename = current_time.strftime("%Y%m%d%H%M%S") + ".png"
    save_path = pic_path + "GBT_Random_Search_" + filename
    plt.savefig(save_path)

    # plt.show()

    # 将Accuracy, Precision, Recall保存到指定Excel中
    ID = "GBT_random_"
    result_df = pd.DataFrame({'ID': [ID],
                              'Accuracy': [accuracy_test],
                              'Precision': [precision_test],
                              'Recall': [recall_test],
                              'F1 Score': [f1],
                              'C': [random_search.best_estimator_],
                              'save_path': [save_path]
                              })
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

    plt.close()
    print("GBT Random is complete")


def AdaBoost_Grid(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y):
    # Load data
    file_path = input_file_path
    sheet_name = sheet_name
    diabetes = pd.read_excel(file_path, sheet_name=sheet_name)

    # Prepare X and y
    X = diabetes[cols_x].values
    y = diabetes[cols_y].values.ravel()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Scale the features
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # Create an 9_AdaBoost classifier
    ada = AdaBoostClassifier(random_state=1)

    param_grid = {
        'n_estimators': [50, 100, 200],  # Number of boosting stages to be run
        'learning_rate': [0.01, 0.1, 0.2],  # Step size at each iteration
    }
    grid_search = GridSearchCV(estimator=ada, param_grid=param_grid, cv=5)
    grid_search.fit(X_train_std, y_train)
    y_pred = grid_search.predict(X_test_std)
    y_pred_prob = grid_search.predict_proba(X_test_std)

    # Get the best 9_AdaBoost model
    best_ada_classifier = grid_search.best_estimator_

    # Use the best model to predict on the test set
    y_pred_test = best_ada_classifier.predict(X_test_std)
    y_pred_prob_test = best_ada_classifier.predict_proba(X_test_std)

    # Calculate accuracy, recall, and precision for the test set
    accuracy_test = accuracy_score(y_test, y_pred_test)
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    precision_test = precision_score(y_test, y_pred_test, average='weighted')
    f1 = f1_score(y_test, y_pred_test, average='weighted')

    # 打印结果
    # print("Best C:", grid_search.best_params_['C'])
    print("Accuracy:", accuracy_test)
    print("Precision:", precision_test)
    print("Recall:", recall_test)
    print("F1 Score:", f1)

    # Calculate ROC curve and AUC for the test set
    fpr_test = dict()
    tpr_test = dict()
    roc_auc_test = dict()

    for i in range(best_ada_classifier.classes_.size):
        fpr_test[i], tpr_test[i], _ = roc_curve(y_test, y_pred_prob_test[:, i], pos_label=i)
        roc_auc_test[i] = auc(fpr_test[i], tpr_test[i])

    # Plotting the ROC curves for the test set
    plt.figure()
    colors = ['#8A2BE2', '#7FFF00', '#FF69B4']  # 蓝紫色、查特鲁斯绿和亮粉红色
    for i, color in zip(range(best_ada_classifier.classes_.size), colors):
        plt.plot(fpr_test[i], tpr_test[i], color=color, lw=2,
                 label='ROC curve (area = %0.2f) for class %s' % (roc_auc_test[i], i))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for AdaBoost')
    plt.legend(loc="lower right")

    # Save the plot with a filename based on the current time
    current_time = datetime.datetime.now()
    filename = current_time.strftime("%Y%m%d%H%M%S") + ".png"
    save_path = pic_path + "AdaBoost_Grid_Search_" + filename
    plt.savefig(save_path)

    # plt.show()

    # 将Accuracy, Precision, Recall保存到指定Excel中
    ID = "AdaBoost_grid_"
    result_df = pd.DataFrame({'ID': [ID],
                              'Accuracy': [accuracy_test],
                              'Precision': [precision_test],
                              'Recall': [recall_test],
                              'F1 Score': [f1],
                              'C': [grid_search.best_estimator_],
                              'save_path': [save_path]
                              })
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

    plt.close()
    print("AdaBoost Grid is complete")


def AdaBoost_Random(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y):
    # Load data
    file_path = input_file_path
    sheet_name = sheet_name
    diabetes = pd.read_excel(file_path, sheet_name=sheet_name)

    # Prepare X and y
    X = diabetes[cols_x].values
    y = diabetes[cols_y].values.ravel()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Scale the features
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # Create an 9_AdaBoost classifier
    ada = AdaBoostClassifier(random_state=1)

    param_dist = {
        'n_estimators': randint(50, 201),  # 在[50, 200]范围内随机选择整数作为n_estimators的值
        'learning_rate': [0.01, 0.1, 0.2],  # Step size at each iteration
    }

    # n_iter参数指定了随机搜索要执行的随机采样次数，可以根据需要设置为合适的值
    random_search = RandomizedSearchCV(ada, param_distributions=param_dist, n_iter=20, cv=5, random_state=1)

    random_search.fit(X_train_std, y_train)

    best_ada_classifier = random_search.best_estimator_
    best_ada_classifier.fit(X_train_std, y_train)

    # 进行预测
    y_pred_test = best_ada_classifier.predict(X_test_std)
    y_pred_prob_test = best_ada_classifier.predict_proba(X_test_std)

    # 计算性能指标
    accuracy_test = accuracy_score(y_test, y_pred_test)
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    precision_test = precision_score(y_test, y_pred_test, average='weighted')
    f1 = f1_score(y_test, y_pred_test, average='weighted')

    # 打印结果
    # print("Best C:", grid_search.best_params_['C'])
    print("Accuracy:", accuracy_test)
    print("Precision:", precision_test)
    print("Recall:", recall_test)
    print("F1 Score:", f1)

    # 计算ROC曲线和AUC
    fpr_test = dict()
    tpr_test = dict()
    roc_auc_test = dict()

    for i in range(best_ada_classifier.classes_.size):
        fpr_test[i], tpr_test[i], _ = roc_curve(y_test, y_pred_prob_test[:, i], pos_label=i)
        roc_auc_test[i] = auc(fpr_test[i], tpr_test[i])

    # Plotting the ROC curves for test set
    plt.figure()
    colors = ['#8A2BE2', '#7FFF00', '#FF69B4']  # 蓝紫色、查特鲁斯绿和亮粉红色
    for i, color in zip(range(best_ada_classifier.classes_.size), colors):
        plt.plot(fpr_test[i], tpr_test[i], color=color, lw=2,
                 label='ROC curve (area = %0.2f) for class %s' % (roc_auc_test[i], i))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for AdaBoost')
    plt.legend(loc="lower right")

    # 将图像根据时间命名并保存到指定路径
    current_time = datetime.datetime.now()
    filename = current_time.strftime("%Y%m%d%H%M%S") + ".png"
    save_path = pic_path + "AdaBoost_Random_Search_" + filename
    plt.savefig(save_path)

    # plt.show()

    # 将Accuracy, Precision, Recall保存到指定Excel中
    ID = "AdaBoost_random_"
    result_df = pd.DataFrame({'ID': [ID],
                              'Accuracy': [accuracy_test],
                              'Precision': [precision_test],
                              'Recall': [recall_test],
                              'F1 Score': [f1],
                              'C': [random_search.best_estimator_],
                              'save_path': [save_path]
                              })
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

    plt.close()
    print("AdaBoost Random is complete")


def XGBoost_Grid(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y):
    # Load data
    file_path = input_file_path
    sheet_name = sheet_name
    diabetes = pd.read_excel(file_path, sheet_name=sheet_name)

    # Prepare X and y
    X = diabetes[cols_x].values
    y = diabetes[cols_y].values.ravel()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Scale the features
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # Create an 10_XGBoost classifier
    xgb = XGBClassifier(random_state=1)

    param_grid = {
        'n_estimators': [50, 100, 200],  # Number of boosting stages to be run
        'learning_rate': [0.01, 0.1, 0.2],  # Step size at each iteration
        'max_depth': [3, 5, 7],  # Maximum depth of the individual trees
    }
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5)
    grid_search.fit(X_train_std, y_train)
    y_pred = grid_search.predict(X_test_std)
    y_pred_prob = grid_search.predict_proba(X_test_std)

    # Get the best 10_XGBoost model
    best_xgb_classifier = grid_search.best_estimator_

    # Use the best model to predict on the test set
    y_pred_test = best_xgb_classifier.predict(X_test_std)
    y_pred_prob_test = best_xgb_classifier.predict_proba(X_test_std)

    # Calculate accuracy, recall, and precision for the test set
    accuracy_test = accuracy_score(y_test, y_pred_test)
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    precision_test = precision_score(y_test, y_pred_test, average='weighted')
    f1 = f1_score(y_test, y_pred_test, average='weighted')

    # 打印结果
    # print("Best C:", grid_search.best_params_['C'])
    print("Accuracy:", accuracy_test)
    print("Precision:", precision_test)
    print("Recall:", recall_test)
    print("F1 Score:", f1)

    # Calculate ROC curve and AUC for the test set
    fpr_test = dict()
    tpr_test = dict()
    roc_auc_test = dict()

    for i in range(best_xgb_classifier.classes_.size):
        fpr_test[i], tpr_test[i], _ = roc_curve(y_test, y_pred_prob_test[:, i], pos_label=i)
        roc_auc_test[i] = auc(fpr_test[i], tpr_test[i])

    # Plotting the ROC curves for the test set
    plt.figure()
    colors = ['#8A2BE2', '#7FFF00', '#FF69B4']  # 蓝紫色、查特鲁斯绿和亮粉红色
    for i, color in zip(range(best_xgb_classifier.classes_.size), colors):
        plt.plot(fpr_test[i], tpr_test[i], color=color, lw=2,
                 label='ROC curve (area = %0.2f) for class %s' % (roc_auc_test[i], i))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for XGBoost')
    plt.legend(loc="lower right")

    # Save the plot with a filename based on the current time
    current_time = datetime.datetime.now()
    filename = current_time.strftime("%Y%m%d%H%M%S") + ".png"
    save_path = pic_path + "XGBoost_Grid_Search_" + filename
    plt.savefig(save_path)

    # plt.show()

    # 将Accuracy, Precision, Recall保存到指定Excel中
    ID = "XGBoost_grid_"
    result_df = pd.DataFrame({'ID': [ID],
                              'Accuracy': [accuracy_test],
                              'Precision': [precision_test],
                              'Recall': [recall_test],
                              'F1 Score': [f1],
                              'C': [grid_search.best_estimator_],
                              'save_path': [save_path]
                              })
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

    plt.close()
    print("XGBoost Grid is complete")


def XGBoost_Random(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y):
    # Load data
    file_path = input_file_path
    sheet_name = sheet_name
    diabetes = pd.read_excel(file_path, sheet_name=sheet_name)

    # Prepare X and y
    X = diabetes[cols_x].values
    y = diabetes[cols_y].values.ravel()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Scale the features
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # Create an 10_XGBoost classifier
    xgb = XGBClassifier(random_state=1)

    param_dist = {
        'n_estimators': randint(50, 201),  # 在[50, 200]范围内随机选择整数作为n_estimators的值
        'learning_rate': [0.01, 0.1, 0.2],  # Step size at each iteration
    }

    # n_iter参数指定了随机搜索要执行的随机采样次数，可以根据需要设置为合适的值
    random_search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=20, cv=5, random_state=1)

    random_search.fit(X_train_std, y_train)

    best_xgb_classifier = random_search.best_estimator_
    best_xgb_classifier.fit(X_train_std, y_train)

    # 进行预测
    y_pred_test = best_xgb_classifier.predict(X_test_std)
    y_pred_prob_test = best_xgb_classifier.predict_proba(X_test_std)

    # 计算性能指标
    accuracy_test = accuracy_score(y_test, y_pred_test)
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    precision_test = precision_score(y_test, y_pred_test, average='weighted')
    f1 = f1_score(y_test, y_pred_test, average='weighted')

    # 打印结果
    # print("Best C:", grid_search.best_params_['C'])
    print("Accuracy:", accuracy_test)
    print("Precision:", precision_test)
    print("Recall:", recall_test)
    print("F1 Score:", f1)

    # 计算ROC曲线和AUC
    fpr_test = dict()
    tpr_test = dict()
    roc_auc_test = dict()

    for i in range(best_xgb_classifier.classes_.size):
        fpr_test[i], tpr_test[i], _ = roc_curve(y_test, y_pred_prob_test[:, i], pos_label=i)
        roc_auc_test[i] = auc(fpr_test[i], tpr_test[i])

    # Plotting the ROC curves for test set
    plt.figure()
    colors = ['#8A2BE2', '#7FFF00', '#FF69B4']  # 蓝紫色、查特鲁斯绿和亮粉红色
    for i, color in zip(range(best_xgb_classifier.classes_.size), colors):
        plt.plot(fpr_test[i], tpr_test[i], color=color, lw=2,
                 label='ROC curve (area = %0.2f) for class %s' % (roc_auc_test[i], i))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for XGBoost')
    plt.legend(loc="lower right")

    # 将图像根据时间命名并保存到指定路径
    current_time = datetime.datetime.now()
    filename = current_time.strftime("%Y%m%d%H%M%S") + ".png"
    save_path = pic_path + "XGBoost_Random_Search_" + filename
    plt.savefig(save_path)

    # plt.show()

    # 将Accuracy, Precision, Recall保存到指定Excel中
    ID = "XGBoost_random_"
    result_df = pd.DataFrame({'ID': [ID],
                              'Accuracy': [accuracy_test],
                              'Precision': [precision_test],
                              'Recall': [recall_test],
                              'F1 Score': [f1],
                              'C': [random_search.best_estimator_],
                              'save_path': [save_path]
                              })
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

    plt.close()
    print("XGBoost Random is complete")


def NB(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y):
    # Load data
    file_path = input_file_path
    sheet_name = sheet_name
    diabetes = pd.read_excel(file_path, sheet_name=sheet_name)

    # Prepare X and y
    X = diabetes[cols_x].values
    y = diabetes[cols_y].values.ravel()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # Scale the features
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # Create a Gaussian Naive Bayes classifier
    nb = GaussianNB()

    # Fit the model on the training data
    nb.fit(X_train_std, y_train)

    # Use the model to predict on the test set
    y_pred_test = nb.predict(X_test_std)
    y_pred_prob_test = nb.predict_proba(X_test_std)

    # Calculate accuracy, recall, and precision for the test set
    accuracy_test = accuracy_score(y_test, y_pred_test)
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    precision_test = precision_score(y_test, y_pred_test, average='weighted')
    f1 = f1_score(y_test, y_pred_test, average='weighted')

    # 打印结果
    # print("Best C:", grid_search.best_params_['C'])
    print("Accuracy:", accuracy_test)
    print("Precision:", precision_test)
    print("Recall:", recall_test)
    print("F1 Score:", f1)

    # Calculate ROC curve and AUC for the test set
    fpr_test = dict()
    tpr_test = dict()
    roc_auc_test = dict()

    for i in range(nb.classes_.size):
        fpr_test[i], tpr_test[i], _ = roc_curve(y_test, y_pred_prob_test[:, i], pos_label=i)
        roc_auc_test[i] = auc(fpr_test[i], tpr_test[i])

    # Plotting the ROC curves for the test set
    plt.figure()
    colors = ['#8A2BE2', '#7FFF00', '#FF69B4']  # 蓝紫色、查特鲁斯绿和亮粉红色
    for i, color in zip(range(nb.classes_.size), colors):
        plt.plot(fpr_test[i], tpr_test[i], color=color, lw=2,
                 label='ROC curve (area = %0.2f) for class %s' % (roc_auc_test[i], i))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Naive Bayes')
    plt.legend(loc="lower right")

    # Save the plot with a filename based on the current time
    current_time = datetime.datetime.now()
    filename = current_time.strftime("%Y%m%d%H%M%S") + ".png"
    save_path = pic_path + "NaiveBayes_" + filename
    plt.savefig(save_path)

    # plt.show()

    # 将Accuracy, Precision, Recall保存到指定Excel中
    ID = "Naive_Bayes"
    result_df = pd.DataFrame({'ID': [ID],
                              'Accuracy': [accuracy_test],
                              'Precision': [precision_test],
                              'Recall': [recall_test],
                              'F1 Score': [f1],
                              'C': [None],
                              'save_path': [save_path]
                              })
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

    plt.close()
    print("NB is complete")

def Prediction_Models(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y):
    LR_Grid(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y)
    LR_Random(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y)

    RF_Grid(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y)
    RF_Random(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y)

    SVM_Grid(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y)
    SVM_Random(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y)

    DT_Grid(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y)
    DT_Random(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y)

    KNN_Grid(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y)
    KNN_Random(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y)

    NN_Grid(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y)
    NN_Random(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y)

    GBT_Grid(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y)
    GBT_Random(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y)

    AdaBoost_Grid(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y)
    AdaBoost_Random(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y)

    XGBoost_Grid(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y)
    XGBoost_Random(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y)

    NB(input_file_path, sheet_name, output_file_path, pic_path, cols_x, cols_y)

    print("==Prediction Models is over==")