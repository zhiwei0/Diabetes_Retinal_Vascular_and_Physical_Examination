import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve, auc
import datetime

diabetes = pd.read_excel("D:/Analysis of diabetes/data_process_normalization.xlsx", sheet_name='中间范围_糖尿病')

# Prepare X and y
# 0-1 所有指标
# X = diabetes[['artery_caliber','vein_caliber','frac','AVR','artery_curvature','vein_curvature','artery_BSTD','vein_BSTD',
#               'artery_simple_curvatures','vein_simple_curvatures','artery_Branching_Coefficient','vein_Branching_Coefficient',
#               'artery_Num1stBa','vein_Num1stBa','artery_Branching_Angle','vein_Branching_Angle','artery_Angle_Asymmetry',
#               'vein_Angle_Asymmetry','artery_LDR','vein_LDR','artery_Asymmetry_Raito','vein_Asymmetry_Raito','artery_JED',
#               'vein_JED','artery_Optimal_Deviation','vein_Optimal_Deviation','vessel_length_density','vessel_area_density']]
# 0-1 筛选出的指标
# X = diabetes[['frac','vein_Num1stBa','artery_Branching_Angle','artery_Asymmetry_Raito']]
# 1-2
X = diabetes[['frac','vein_Branching_Angle','artery_Optimal_Deviation','vessel_length_density']]

y = diabetes['Result'].replace({1:0,2:1})
# y = diabetes['Result'].values.ravel()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Scale the features
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Create Decision Tree model object with default hyperparameters
dt = DecisionTreeClassifier()

# Train the Decision Tree model on the training data
dt.fit(X_train_std, y_train)

# Make predictions on the test set
y_pred_test = dt.predict(X_test_std)
y_pred_prob_test = dt.predict_proba(X_test_std)

# Calculate performance metrics and ROC curve
accuracy_test = accuracy_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test, average='weighted')
precision_test = precision_score(y_test, y_pred_test, average='weighted')

fpr_test = dict()
tpr_test = dict()
roc_auc_test = dict()

for i in range(dt.classes_.size):
    fpr_test[i], tpr_test[i], _ = roc_curve(y_test, y_pred_prob_test[:, i], pos_label=i)
    roc_auc_test[i] = auc(fpr_test[i], tpr_test[i])

# Print results and plot ROC curve
print("Accuracy:", accuracy_test)
print("Recall:", recall_test)
print("Precision:", precision_test)

plt.figure()
colors = ['blue', 'red', 'green']
for i, color in zip(range(dt.classes_.size), colors):
    plt.plot(fpr_test[i], tpr_test[i], color=color, lw=2, label='ROC curve (area = %0.2f) for class %s' % (roc_auc_test[i], i))
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Test Set')
plt.legend(loc="lower right")

current_time = datetime.datetime.now()
filename = current_time.strftime("%Y%m%d%H%M%S") + ".png"
save_path = "D:/Analysis of diabetes/ROC/" + "Decision_Tree_Default_Hyperparameters_" + filename
plt.savefig(save_path)

plt.show()
