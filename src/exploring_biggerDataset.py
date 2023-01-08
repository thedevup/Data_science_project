import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
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

# Normalize Features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

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
undersample = RandomUnderSampler(sampling_strategy='majority')
# fit and apply the transform
train_data, train_label = undersample.fit_resample(train_data, train_label)
# summarize class distribution
print(Counter(train_label))

def logistic_regression():
    # Train the logistic regression model
    model = LogisticRegression(max_iter=3000)
    model.fit(train_data, train_label)
    # Make predictions on new data
    y_pred = model.predict(test_data)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(test_label, y_pred)
    print("Logistic regression accuracy: : %.2f" % (accuracy*100))
    print(metrics.classification_report(test_label,y_pred))

def knn_model():
    neigh = KNeighborsClassifier(n_neighbors=7)


    # Training the model
    neigh.fit(train_data, train_label)
    # Make predictions on new data
    y_pred = neigh.predict(test_data)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(test_label, y_pred)
    print("K-nn accuracy: %.2f" %(accuracy*100))    
    print(metrics.classification_report(test_label,y_pred))

def naive_bayes():
    nb = GaussianNB()

    # Training the model
    nb.fit(train_data, train_label)

    # Make predictions on new data
    y_pred = nb.predict(test_data)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(test_label, y_pred)
    print("Naive Bayes accuracy: %.2f" %(accuracy*100))
    print(metrics.classification_report(test_label,y_pred))
    

def svm_model():
    svm = SVC()

    # Training the model
    svm.fit(train_data, train_label)

    # Make predictions on new data
    y_pred = svm.predict(test_data)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(test_label, y_pred)
    print("SVM accuracy: %.2f' " %(accuracy*100))
    print(metrics.classification_report(test_label,y_pred))

def decision_tree():
    dt = DecisionTreeClassifier()

    # Training the model
    dt.fit(train_data, train_label)

    # Make predictions on new data
    y_pred = dt.predict(test_data)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(test_label, y_pred)
    print('Decision_Tree accuracy: %.2f' % (accuracy*100))
    print(metrics.classification_report(test_label,y_pred))

logistic_regression()
naive_bayes()
svm_model()
decision_tree()
knn_model()
