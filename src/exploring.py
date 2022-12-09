import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

#Loading data
file_errors_location = "D:\\University\\FourthYear\\SecondTerm\\DataScience\\Data\\SelfAssessmentAndTestCenter.xlsx"
df = pd.read_excel(file_errors_location)
print(df)

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
df= df.dropna()

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

#select columns with index positions 1 and 3
a = df.iloc[:,[1,3]]
print(a)
b = df.iloc[:,5:]
print(b)

updated_data =pd.concat([a,b],axis=1)
print(updated_data)

#splitting target from data
X = updated_data
Y = df.iloc[:,4]

# Standardizing the features, because bmi is in higher range compared to likert scale 
X= StandardScaler().fit_transform(X)
print("After standardizing",X)

# Creating a PCA object with 0.99 as the target explained variance
pca = PCA(n_components=0.99)

# Fiting the PCA model to the data
pca.fit(X)

# Transforming the data using the fitted PCA model
X_pca = pca.transform(X)

# Creating a new dataframe from the transformed data
column_names = ['PCA_' + str(i+1) for i in range(X_pca.shape[1])]
df_pca = pd.DataFrame(data=X_pca, columns=column_names)
print(df_pca)

#splitting data into train, valid and test data
train_data, test_data, train_label, test_label = train_test_split(df_pca, Y, test_size=0.05, random_state=1)
train_data, valid_data, train_label, valid_label = train_test_split(train_data, train_label, test_size=0.15, random_state=1)

def logistic_regression():
    # Train the logistic regression model
    model = LogisticRegression(max_iter=3000)
    model.fit(train_data, train_label)

    # Make predictions on new data
    y_pred = model.predict(test_data)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(test_label, y_pred)
    print(accuracy)

def knn_model():
    # Setup arrays to store train and test accuracies
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
    plt.show()

def naive_bayes():
    nb = GaussianNB()

    # Training the model
    nb.fit(train_data, train_label)

    # Make predictions on new data
    y_pred = nb.predict(test_data)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(test_label, y_pred)
    print(accuracy)

def svm_model():
    svm = SVC()

    # Training the model
    svm.fit(train_data, train_label)

    # Make predictions on new data
    y_pred = svm.predict(test_data)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(test_label, y_pred)
    print(accuracy)

def decision_tree():
    dt = DecisionTreeClassifier()

    # Training the model
    dt.fit(train_data, train_label)

    # Make predictions on new data
    y_pred = dt.predict(test_data)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(test_label, y_pred)
    print('Test Accuracy: %.2f' % (accuracy*100))

logistic_regression()
naive_bayes()
svm_model()
decision_tree()
    




