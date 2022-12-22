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

#Loading data
file_errors_location = "D:\\University\\FourthYear\\SecondTerm\\DataScience\\Data\\SelfAssessmentAndTestCenter.xlsx"
df = pd.read_excel(file_errors_location)
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

# Calculating BMI from weight and height
# Formula : ( weight (kg) / height (cm) / height (cm) )x 10,000
df['bmi'] = ((df['X109_08']/df['X109_07']/df['X109_07'])*10000).round(2)

# Some rows for both weight and height have the value 0
df.loc[df['X109_07'] == 0]
df.loc[df['X109_08'] == 0]


# That results in some rows in bmi having the values 0 or inf
print("bmi",df['bmi'])

# We remove rows with the value 'inf' or 0 from bmi
df= df[df['bmi'] != float("inf")]
df = df[df['bmi'] != 0]

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

# Standardizing the features, because bmi is in higher range compared to likert scale 
X= MinMaxScaler().fit_transform(X)
print("After standardizing",X)

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
print(df)

#-----

# Creating a new dataframe from the transformed data
column_names = ['PCA_' + str(i+1) for i in range(X_pca.shape[1])]
df_pca = pd.DataFrame(data=X_pca, columns=column_names)
print(df_pca)

#splitting data into train, valid and test data
train_data, test_data, train_label, test_label = train_test_split(df_pca, Y, test_size=0.10, random_state=1)
train_data, valid_data, train_label, valid_label = train_test_split(train_data, train_label, test_size=0.15, random_state=1)

def logistic_regression():
    # Train the logistic regression model
    model = LogisticRegression(max_iter=3000)
    model.fit(train_data, train_label)

    # get importance
    importance = model.coef_[0]
    # summarize feature importance
    for i,v in enumerate(importance):
     print("feature:", df_pca.columns[i],"Importance score: ",format(v, '.5f'))
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()

    # Make predictions on new data
    y_pred = model.predict(test_data)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(test_label, y_pred)
    print("Logistic regression accuracy: : %.2f" % (accuracy*100))
    print(metrics.classification_report(test_label,y_pred))

def knn_model():
    '''# Setup arrays to store train and test accuracies
    neighbors = np.arange(1, 9)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    # Loop over different values of k
    for i, k in enumerate(neighbors):
        # Setup a k-NN Classifier with k neighbors: knn
        knn = KNeighborsClassifier(n_neighbors=k)

        # Fit the classifier to the training data
        knn.fit(train_data, train_label)
        
        #Compute accuracy on the training set
        train_accuracy[i] = knn.score(train_data, train_label)

        #Compute accuracy on the testing set
        test_accuracy[i] = knn.score(test_data, test_label)

    # Generate plot
    plt.title('k-NN: Varying Number of Neighbors')
    plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
    plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
    plt.legend()
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.show()'''
    neigh = KNeighborsClassifier(n_neighbors=7)


    # Training the model
    neigh.fit(train_data, train_label)
   
    # perform permutation importance
    results = permutation_importance(neigh,train_data, train_label, scoring='accuracy')
    # get importance
    importance = results.importances_mean
    # summarize feature importance
    for i,v in enumerate(importance):
       #print('Feature: %0d, Score: %.5f' % (i,v))
       print("feature:", df_pca.columns[i],"Importance score: ",format(v, '.5f'))
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()

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

    # perform permutation importance
    results = permutation_importance(nb,train_data, train_label, scoring='accuracy')
    # get importance
    importance = results.importances_mean
    # summarize feature importance
    for i,v in enumerate(importance):
       print("feature:", df_pca.columns[i],"Importance score: ",format(v, '.5f'))
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()

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

    # get importance
    importance = svm.coef_[0]
    # summarize feature importance
    for i,v in enumerate(importance):
     print("feature:", df_pca.columns[i],"Importance score: ",format(v, '.5f'))
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()

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

    # get importance
    importance = dt.feature_importances_
    # summarize feature importance
    for i,v in enumerate(importance):
       print("feature:", df_pca.columns[i],"Importance score: ",format(v, '.5f'))
    # plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()

    # Make predictions on new data
    y_pred = dt.predict(test_data)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(test_label, y_pred)
    print('Decision_Tree accuracy: %.2f' % (accuracy*100))
    print(metrics.classification_report(test_label,y_pred))

'''logistic_regression()
naive_bayes()
svm_model()
decision_tree()'''
knn_model()

    




