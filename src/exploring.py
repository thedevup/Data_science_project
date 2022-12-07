import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


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
'''
#correlations
print("\nCORRELATIONS:")
corr=(df[3:].corr())
print(corr)
sns.heatmap(df[3:].corr(),annot=True)
plt.show()'''

#select columns with index positions 1 and 3
a = df.iloc[:,[1,3]]
print(a)
b = df.iloc[:,6:]
print(b)

updated_data =pd.concat([a,b],axis=1)
print(updated_data)

#splitting target from data
X = updated_data
Y = df.iloc[:,5]


#splitting data into train, valid and test data
train_data, test_data, train_label, test_label = train_test_split(X, Y, test_size=0.1, random_state=1)
train_data, valid_data, train_label, valid_label = train_test_split(train_data, train_label, test_size=0.15, random_state=1)

#k-nearest neighbor model
# build a k-NN model with k=3, k nearest model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_data, train_label)

# evaluate predictions
train_accuracy = knn.score(train_data, train_label)
print('Accuracy: %.2f' % (train_accuracy*100))

                              
test_accuracy = knn.score(test_data, test_label)
print('Accuracy: %.2f' % (test_accuracy*100))

