import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

#Loading data
file_errors_location = "D:\\University\\FourthYear\\SecondTerm\\DataScience\\Data\\TS2018-2019_AISS (AM students).xlsx"
df = pd.read_excel(file_errors_location)
df.info()
    
#select columns with index positions 1 and 3
a = df.iloc[:,[1,3]]
b = df.iloc[:,6:]

updated_data =pd.concat([a,b],axis=1)
print(updated_data)

#splitting target from data
X = updated_data.values
Y = df.iloc[:,5]

print('0 and 1 count in finals: ',df['Finals'].value_counts())
X_col = updated_data
print(X)
print(Y)

# Creating a PCA object with 0.99 as the target explained variance
pca = PCA(n_components=0.99).fit(X)

# Fiting the PCA model to the data
#pca.fit(X)

# Transforming the data using the fitted PCA model
X_pca = pca.transform(X)
#------

# number of components
n_pca= pca.components_.shape[0]

# get the index of the most important feature on EACH component
# LIST COMPREHENSION HERE
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pca)]

initial_feature_names = X_col.columns
# get the names
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pca)]

# LIST COMPREHENSION HERE AGAIN
dic = {'PCA{}'.format(i+1): most_important_names[i] for i in range(n_pca)}

# build the dataframe
df = pd.DataFrame(dic.items())
print(df[0:50])

#-----
# Creating a new dataframe from the transformed data
column_names = ['PCA_' + str(i+1) for i in range(X_pca.shape[1])]
df_pca = pd.DataFrame(data=X_pca, columns=column_names)
print(df_pca)

#splitting data into train, valid and test data
train_data, test_data, train_label, test_label = train_test_split(df_pca, Y, test_size=0.10, random_state=1)
train_data, valid_data, train_label, valid_label = train_test_split(train_data, train_label, test_size=0.15, random_state=1)

# define oversampling strategy
#oversample = RandomOverSampler(sampling_strategy='minority')
# fit and apply the transform
#train_data, train_label = oversample.fit_resample(train_data, train_label)

# define undersample strategy
#undersample = RandomUnderSampler(sampling_strategy='majority')
# fit and apply the transform
#train_data, train_label = undersample.fit_resample(train_data, train_label)
# summarize class distribution
print(Counter(train_label))

def optimizing_LR():
    # define models and parameters
    model = LogisticRegression()
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['l2']
    c_values = [100, 10, 1.0, 0.1, 0.01]
    # define grid search
    grid = dict(solver=solvers,penalty=penalty,C=c_values)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(train_data, train_label)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    #Following result was obtained
    #Best: 0.601450 using {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear'}

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
    #'C': 0.1, 'gamma': 'scale', 'kernel': 'sigmoid'}

def optimizing_gaussiannb():
    model=GaussianNB()
    params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}
    grid_search = GridSearchCV(estimator=model, param_grid=params_NB, scoring='accuracy',error_score=0) 
    grid_result=grid_search.fit(train_data, train_label)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    #Result obtained
    #'var_smoothing': 0.8111308307896871

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
    #'criterion': 'entropy', 'max_depth': 5, 'min_samples_leaf': 100
    
def optimizing_knn():
    model =KNeighborsClassifier()
    params = {'n_neighbors': [1,2,3,4,5,6,7,8,9,10]}
    grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy',error_score=0) 
    grid_result=grid_search.fit(train_data, train_label)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    #Result obtained
    #'criterion': 'entropy', 'max_depth': 5, 'min_samples_leaf': 100
    
def logistic_regression():
    # Train the logistic regression model
    model = LogisticRegression(C=0.1,penalty='l2',solver='liblinear')
    model.fit(train_data, train_label)
    # Make predictions on new data
    y_pred = model.predict(test_data)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(test_label, y_pred)
    print("Logistic regression accuracy: : %.2f" % (accuracy*100))
    print(metrics.classification_report(test_label,y_pred))
    cm = confusion_matrix(test_label, y_pred, labels = model.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
    display.plot()
    plt.show()

def knn_model():
    model = KNeighborsClassifier(n_neighbors=5)

    # Training the model
    model.fit(train_data, train_label)
    # Make predictions on new data
    y_pred = model.predict(test_data)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(test_label, y_pred)
    print("K-nn accuracy: %.2f" %(accuracy*100))    
    print(metrics.classification_report(test_label,y_pred))
    cm = confusion_matrix(test_label, y_pred, labels = model.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
    display.plot()
    plt.show()

def naive_bayes():
    model = GaussianNB(var_smoothing=0.8111308307896871)

    # Training the model
    model.fit(train_data, train_label)

    # Make predictions on new data
    y_pred = model.predict(test_data)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(test_label, y_pred)
    print("Naive Bayes accuracy: %.2f" %(accuracy*100))
    print(metrics.classification_report(test_label,y_pred))
    cm = confusion_matrix(test_label, y_pred, labels = model.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
    display.plot()
    plt.show()
    

def svm_model():
    model = SVC(C=0.1, gamma= 'scale', kernel= 'sigmoid')

    # Training the model
    model.fit(train_data, train_label)

    # Make predictions on new data
    y_pred = model.predict(test_data)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(test_label, y_pred)
    print("SVM accuracy: %.2f' " %(accuracy*100))
    print(metrics.classification_report(test_label,y_pred))
    cm = confusion_matrix(test_label, y_pred, labels = model.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
    display.plot()
    plt.show()

def decision_tree():
    model = DecisionTreeClassifier(criterion= 'entropy', max_depth= 5, min_samples_leaf= 100)

    # Training the model
    model.fit(train_data, train_label)

    # Make predictions on new data
    y_pred = model.predict(test_data)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(test_label, y_pred)
    print('Decision_Tree accuracy: %.2f' % (accuracy*100))
    print(metrics.classification_report(test_label,y_pred))
    cm = confusion_matrix(test_label, y_pred, labels = model.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
    display.plot()
    plt.show()

#optimizing_LR()
#optimizing_SVM()
#optimizing_gaussiannb()
#optimizing_decisiontree()
#optimizing_knn()

logistic_regression()
naive_bayes()
svm_model()
decision_tree()
knn_model()

