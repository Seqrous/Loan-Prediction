import pandas as pd
import numpy as np

dataset = pd.read_csv('dataset.csv')

dataset.info()
dataset.isnull().sum()

""" PREPROCESSING """
# Splitting into categorical and numerical data

cat_data = dataset.iloc[:, 0:6].values
cat_data = np.append(cat_data, dataset.iloc[:, 11:13].values, axis = 1)

num_data = dataset.iloc[:, 6:11].values

cat_data = pd.DataFrame(cat_data)
num_data = pd.DataFrame(num_data)

# Handling the missing numerical data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = 0, strategy = 'mean')
num_data.values[:, 0:2] = imputer.fit_transform(num_data.values[:, 0:2])

num_data[2].fillna(num_data.iloc[:, 2].mean(), inplace = True)

imputer_most_frequent = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
num_data.values[:, 3:5] = imputer_most_frequent.fit_transform(num_data.values[:, 3:5])

# Handling the missing categorical data
cat_data.values[:, :] = imputer_most_frequent.fit_transform(cat_data.values[:, :])

# Check if there are any nulls left
cat_data.isnull().sum().any()
num_data.isnull().sum().any()

# Encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
label_encoder = LabelEncoder()
cat_data.values[:, 2] = label_encoder.fit_transform(cat_data.values[:, 2])
cat_data.values[:, 4] = label_encoder.fit_transform(cat_data.values[:, 4])
cat_data.values[:, 5] = label_encoder.fit_transform(cat_data.values[:, 5])
cat_data.values[:, 6] = label_encoder.fit_transform(cat_data.values[:, 6])
cat_data.values[:, 7] = label_encoder.fit_transform(cat_data.values[:, 7])
cat_data[3] = cat_data[3].replace({"3+": "3"})

# Defining dependent variable
y = cat_data.iloc[:, 7]
y = y.astype(float)
# Dropping unnecessary data - gender, id and dependent variable
cat_data.drop(labels = [0, 1, 7], axis = 1, inplace = True)
# Joining categorical and numerical data
dataset = pd.concat([num_data, cat_data], axis = 1)
# Reset indices of columns
dataset.columns = range(dataset.shape[1])

# Encoding Propety Area using OneHotEncoder
transformer = ColumnTransformer([('Property Area', OneHotEncoder(), [9])], remainder = 'passthrough')
X = np.array(transformer.fit_transform(dataset), dtype = np.float64)
# Drop one dummy variable
X = X[:, 1:]
# Splitting data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 12)

# Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

""" CLASSIFICATION """

"""" Logistic Regression """
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state = 42, n_jobs = -1)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

from sklearn.metrics import confusion_matrix, precision_score, recall_score
log_reg_cm = confusion_matrix(y_test, y_pred)
log_reg_precision = precision_score(y_test, y_pred)
# 0.8(3)
recall_score(y_test, y_pred)

from sklearn.metrics import f1_score
log_reg_score = f1_score(y_test, y_pred)
# 0.8839779...

""" K-NN """
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', n_jobs = -1)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

knn_cm = confusion_matrix(y_test, y_pred)
knn_precision = precision_score(y_test, y_pred)
# 0.82105...
knn_score = f1_score(y_test, y_pred)
# 0.8(6)

""" SVM Linear Kernel """
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

svm_cm = confusion_matrix(y_test, y_pred)
svm_precision = precision_score(y_test, y_pred)
# 0.(81)
svm_score = f1_score(y_test, y_pred)
# 0.880434...

""" SVM RBF Kernel"""
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

svm_RBF_cm = confusion_matrix(y_test, y_pred)
svm_RBF_precision = precision_score(y_test, y_pred)
# 0.(81)
svm_RBF_score = f1_score(y_test, y_pred)
# 0.880434...

""" Naive Bayes """
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

naive_bayes_cm = confusion_matrix(y_test, y_pred)
naive_bayes_precision = precision_score(y_test, y_pred)
# 0.(81)
naive_bayes_score = f1_score(y_test, y_pred)
# 0.880434...

""" Decision Tree """
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

decision_tree_cm = confusion_matrix(y_test, y_pred)
decision_tree_precision = precision_score(y_test, y_pred)
# 0.838235...
decision_tree_score = f1_score(y_test, y_pred)
# 0.745098...

""" Random Forest """
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

forest_cm = confusion_matrix(y_test, y_pred)
forest_precision = precision_score(y_test, y_pred)
# 0.8(3)
forest_score = f1_score(y_test, y_pred)
# 0.828402...

""" SUMMARY """
# Choosing best Classifier
scores = [['Logistic Regression', log_reg_precision, log_reg_score],
          ['K-NN', knn_precision, knn_score],
          ['SVM Linear Kernel', svm_precision, svm_score],
          ['SVM RBF Kernel', svm_RBF_precision, svm_RBF_score],
          ['Naive Bayes', naive_bayes_precision, naive_bayes_score],
          ['Decision Tree', decision_tree_precision, decision_tree_score],
          ['Random Forest', forest_precision, forest_score]]

scores_list = []
for i in range(len(scores)):
    scores_list.append(scores[i][2])
best_score = max(scores_list)
best_classifier_name = scores[scores_list.index(best_score)][0]
best_classifier_precision = scores[scores_list.index(best_score)][1]

print('The best classifier is ', best_classifier_name, ' with the score: ', best_score, ' and precision: ', best_classifier_precision)

# Logistic Regression seems to be the most effective most of the time
# Cross validation to verify whether it's overfitting or not and to make sure it performs well
from sklearn.model_selection import cross_val_score
cross_val_scores = cross_val_score(log_reg, X_train, y_train, cv = 10)
print("Scores: ", cross_val_scores)
print("Mean: ", cross_val_scores.mean())
print("STD: ", cross_val_scores.std())

# Grid Search
from sklearn.model_selection import GridSearchCV
param_grid = [
        {'penalty': ['l1', 'l2'], 'C': [0.25, 0.5, 1, 2]}
    ]
log_reg_fin = LogisticRegression()
grid_search = GridSearchCV(log_reg_fin, param_grid, cv = 5, scoring = 'neg_mean_squared_error')
grid_search.fit(X_train, y_train)

grid_search.best_params_
grid_search.best_estimator_

cv_res = grid_search.cv_results_
for mean_score, params in zip(cv_res['mean_test_score'], cv_res['params']):
    print(np.sqrt(-mean_score), params)
