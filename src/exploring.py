import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.inspection import permutation_importance
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

#Loading data
file_errors_location = "../Data/SelfAssessmentAndTestCenter.xlsx"
df = pd.read_excel(file_errors_location)
df.info()
#replacing 0 with nan
df['X108_03'] = df['X108_03'].replace(0, np.nan)
df['X108_04'] = df['X108_04'].replace(0, np.nan)
df['X109_07'] = df['X109_07'].replace(0, np.nan)
df['X109_08'] = df['X109_08'].replace(0, np.nan)
df=df.fillna(df.mean())
print(df)

#changing to likert scale
conditions = [
    (df['X108_04'] <=20),
    ((df['X108_04'] > 20) & (df['X108_04'] <=40) ),
    ((df['X108_04'] > 40) & (df['X108_04'] <=60)),
    ((df['X108_04'] > 60) & (df['X108_04'] <=80)),
    (df['X108_04'] > 80)
    ]

# create a list of the values we want to assign for each shooting power
values = [1,2,3,4,5]

# create a new column and use np.select to assign values to it using our lists as arguments
df['shooting_power'] = np.select(conditions, values)
print(df[0:5])

#dropping unused data
df.drop(['X108_04'], axis=1, inplace=True)

# Some rows for both weight and height have the value 0
df.loc[df['X109_07'] == 0]
df.loc[df['X109_08'] == 0]

df.drop(['X109_07'], axis=1, inplace=True)
df.drop(['X109_08'], axis=1, inplace=True)

#select columns with index positions 1 and 3
a = df.iloc[:,[1,3]]
print(a)
b = df.iloc[:,6:]
print(b)

updated_data =pd.concat([a,b],axis=1)
print(updated_data)

#splitting target from data
X = updated_data.values
Y = df.iloc[:,5]

X_col = updated_data

# Creating a PCA object with 0.99 as the target explained variance
pca = PCA(n_components=0.99).fit(X)
# Transforming the data using the fitted PCA model
X_pca = pca.transform(X)

#------
# number of components
#n_pca= pca.components_.shape[0]

# get the index of the most important feature on EACH component
# LIST COMPREHENSION HERE
#most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pca)]

"""
initial_feature_names = X_col.columns
# get the names
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pca)]

# -------- Test with and without PCA

# LIST COMPREHENSION HERE AGAIN
dic = {'PCA{}'.format(i+1): most_important_names[i] for i in range(n_pca)}

# build the dataframe
df = pd.DataFrame(dic.items())
print(df[50:80])"""

#-----

# Creating a new dataframe from the transformed data
column_names = ['PCA_' + str(i+1) for i in range(X_pca.shape[1])]
df_pca = pd.DataFrame(data=X_pca, columns=column_names)
#print(df_pca)

#splitting data into train and test sets
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

for train_index, test_index in skf.split(df_pca, Y):
    train_data, test_data = df_pca.iloc[train_index], df_pca.iloc[test_index]
    train_label, test_label = Y.iloc[train_index], Y.iloc[test_index]

# define oversampling strategy
#oversample = RandomOverSampler(sampling_strategy='minority')
# fit and apply the transform
#train_data, train_label = oversample.fit_resample(train_data, train_label)

# Test with over and under sampling

# define undersample strategy
undersample = RandomUnderSampler(sampling_strategy='majority')
# fit and apply the transform
train_data, train_label = undersample.fit_resample(train_data, train_label)

# summarize class distribution
#print(Counter(train_label))


def best_param_lr():
    error_rate = []
    #randmoly start with default params, default value is 1
    c_values=[0.01, 0.1, 1.0,10,100]
    for i in c_values:
     dt = LogisticRegression(C=i)
     dt.fit(train_data,train_label)
     pred_i = dt.predict(test_data)
     error_rate.append(np.mean(pred_i != test_label))
    #plt.figure(figsize=(10,6))
    plt.plot(c_values,error_rate,color='blue', linestyle='dashed',marker='o',markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. c_values Value')
    plt.xlabel('c_values')
    plt.ylabel('Error Rate')
    plt.show()
    print("Minimum error:",min(error_rate),"at c_values =",error_rate.index(min(error_rate)))
    #0.21428571428571427 at c_values = 2

def optimizing_LR():
    # define models and parameters
    model = LogisticRegression()
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['l2']
    c_values = [100, 10, 1.0, 0.1, 0.01]
    # define grid search
    grid = dict(solver=solvers,penalty=penalty,C=c_values)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=5, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(train_data, train_label)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    #Following result was obtained
    #'C': 1.0, 'penalty': 'l2', 'solver': 'newton-cg'

def optimizing_SVM():
    # define model and parameters
    model = SVC()
    kernel = ['poly', 'rbf', 'sigmoid']
    C = [50, 10, 1.0, 0.1, 0.01]
    gamma = ['scale']
    # define grid search
    grid = dict(kernel=kernel,C=C,gamma=gamma)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(train_data, train_label)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    #result obtained after optimizing parameters
    #'C': 1.0, 'gamma': 'scale', 'kernel': 'sigmoid'}

def best_param_nb():
    error_rate = []
    nb_smoothing = np.logspace(0,-9, num=100)
    accuracy_scores = np.zeros(nb_smoothing.shape[0])
    for i in range(nb_smoothing.shape[0]):
     nb = GaussianNB(var_smoothing=nb_smoothing[i])
     nb.fit(train_data,train_label)
     pred_i = nb.predict(test_data)
     error_rate.append(np.mean(pred_i != test_label))
     # Evaluate the accuracy of the model
     accuracy = accuracy_score(test_label, pred_i)
     accuracy_scores[i] = accuracy
    best_index = np.argmax(accuracy_scores)

    # Print the best value of the hyperparameter and its corresponding accuracy
    print("Best variance smoothing value: ", nb_smoothing[best_index])
    print("Accuracy: ", accuracy_scores[best_index])

    #plt.figure()
    plt.plot(nb_smoothing,error_rate,color='blue', linestyle='dashed',marker='o',markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs.  var_Smoothing')
    plt.xlabel('Var_Smoothing')
    plt.ylabel('Error Rate')
    plt.show()
    print("Minimum error:",min(error_rate),"at Var_Smoothing =",error_rate.index(min(error_rate)))
    """ Best variance smoothing value:  0.43287612810830584
    Accuracy:  0.75
    Minimum error: 0.25 at K = 4"""

    #Best variance smoothing value:  1.0

def optimizing_gaussiannb():
    model=GaussianNB()
    params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}
    grid_search = GridSearchCV(estimator=model, param_grid=params_NB, scoring='accuracy',error_score=0)
    grid_result=grid_search.fit(train_data, train_label)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    #Result obtained
    #'var_smoothing': 1.0

def best_param_dt():
    #trying error rate per param in decision tree
    error_rate = []
    #randmoly start with default params,default is none
    #with default valu none error rate =0.5
    max_depth=[1,2,3,4,5,6,7,8,9,10,20]
    for i in max_depth:
     dt = DecisionTreeClassifier(max_depth=i)
     dt.fit(train_data,train_label)
     pred_i = dt.predict(test_data)
     error_rate.append(np.mean(pred_i != test_label))
    #plt.figure(figsize=(10,6))
    plt.plot(max_depth,error_rate,color='blue', linestyle='dashed',marker='o',markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. max_depth Value')
    plt.xlabel('max_depth')
    plt.ylabel('Error Rate')
    plt.show()
    print("Minimum error:",min(error_rate),"at max_depth =",error_rate.index(min(error_rate)))
    #2 was Minimum error: 0.14285714285714285


    #default min_sample_leaf is 1, so trying leaf number around it
    """error_rate = []
    #randmoly start with default params
    min_samples_leaf=[1,2,3,4,5,6,7,8,9,10,20,25]
    for i in min_samples_leaf:
     dt = DecisionTreeClassifier(min_samples_leaf=i)
     dt.fit(train_data,train_label)
     pred_i = dt.predict(test_data)
     error_rate.append(np.mean(pred_i != test_label))
    #plt.figure(figsize=(10,6))
    plt.plot(min_samples_leaf,error_rate,color='blue', linestyle='dashed',marker='o',markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. min_samples_leaf Value')
    plt.xlabel('min_samples_leaf')
    plt.ylabel('Error Rate')
    plt.show()
    print("Minimum error:",min(error_rate),"at min_samples_leaf =",error_rate.index(min(error_rate)))
    #Minimum error: 0.2857142857142857 at min_samples_leaf = 5"""


def optimizing_decisiontree():
    model =DecisionTreeClassifier()
    params = {'max_depth': [2, 3, 5, 10, 20],
              'min_samples_leaf': [5, 10, 20, 50, 100],
              'criterion': ["gini", "entropy"]}
    grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy',error_score=0)
    grid_result=grid_search.fit(train_data, train_label)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    #Result obtained
    #'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 5

def optimizing_knn():
    model =KNeighborsClassifier()
    params = {'n_neighbors': [1,2,3,4,5,6,7,8,9,10]}
    grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy',error_score=0)
    grid_result=grid_search.fit(train_data, train_label)
   # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    #Result obtained
    #n_neighbors:7

def logistic_regression():
    # Train the logistic regression model
    model = LogisticRegression(C=1.0,penalty='l2',solver='newton-cg')
    model.fit(train_data, train_label)
    # Make predictions on new data
    y_pred = model.predict(test_data)

    # get importance
    importance = model.coef_[0]
    # summarize feature importance
    for i,v in enumerate(importance):
     print("feature:", df_pca.columns[i],"Importance score: ",format(v, '.5f'))
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()

    # Calculate the accuracy of the model
    precision = precision_score(test_label, y_pred)
    recall = recall_score(test_label, y_pred)
    mcc = matthews_corrcoef(test_label, y_pred)
    print("Logistic regression precision: : %.2f" % (precision*100))
    print("Logistic regression recall: : %.2f" % (recall*100))
    print("Logistic regression mcc: : %.2f" % (mcc*100))
    print(metrics.classification_report(test_label,y_pred))

def knn_model():
    neigh = KNeighborsClassifier(n_neighbors=7)
    # Training the model
    neigh.fit(train_data, train_label)
    # Make predictions on new data
    y_pred = neigh.predict(test_data)
    '''
    # perform permutation importance
    results = permutation_importance(neigh,train_data, train_label, scoring='accuracy')
    # get importance
    importance = results.importances_mean
    # summarize feature importance
    for i,v in enumerate(importance):
       print("feature:", df_pca.columns[i],"Importance score: ",format(v, '.5f'))
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()
    '''
    # Calculate the accuracy of the model
    precision = precision_score(test_label, y_pred)
    recall = recall_score(test_label, y_pred)
    mcc = matthews_corrcoef(test_label, y_pred)
    print("KNN precision: : %.2f" % (precision*100))
    print("KNN recall: : %.2f" % (recall*100))
    print("KNN mcc: : %.2f" % (mcc*100))
    print(metrics.classification_report(test_label,y_pred))

def naive_bayes():
    nb = GaussianNB(var_smoothing=1.0)

    # Training the model
    nb.fit(train_data, train_label)
    # Make predictions on new data
    y_pred = nb.predict(test_data)

    # Calculate the accuracy of the model
    precision = precision_score(test_label, y_pred)
    recall = recall_score(test_label, y_pred)
    mcc = matthews_corrcoef(test_label, y_pred)
    print("Naive bayes precision: : %.2f" % (precision*100))
    print("Naive bayes recall: : %.2f" % (recall*100))
    print("Naive bayes mcc: : %.2f" % (mcc*100))
    print(metrics.classification_report(test_label,y_pred))

def decision_tree():
    dt = DecisionTreeClassifier(criterion= 'gini', max_depth= 2, min_samples_leaf= 5)

    # Training the model
    dt.fit(train_data, train_label)

    '''
    # get importance
    importance = dt.feature_importances_
    # summarize feature importance
    for i,v in enumerate(importance):
       print("feature:", df_pca.columns[i],"Importance score: ",format(v, '.5f'))
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()
    '''
    # Make predictions on new data
    y_pred = dt.predict(test_data)

    # Calculate the accuracy of the model
    precision = precision_score(test_label, y_pred)
    recall = recall_score(test_label, y_pred)
    mcc = matthews_corrcoef(test_label, y_pred)
    print("Decision tree precision: : %.2f" % (precision*100))
    print("Decision tree recall: : %.2f" % (recall*100))
    print("Decision tree mcc: : %.2f" % (mcc*100))
    print(metrics.classification_report(test_label,y_pred))


#optimizing_LR()
#optimizing_gaussiannb()
#optimizing_decisiontree()
#optimizing_knn()
#best_param_nb()
best_param_dt()
#best_param_lr()

#naive_bayes()
decision_tree()
#logistic_regression()
#knn_model()

'''
logistic_regression()
naive_bayes()
decision_tree()
knn_model()'''
