import pandas as pd
import numpy as np

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

