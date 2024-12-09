import pandas as pd
import numpy as np
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import norm
df = pd.read_csv('ObesityDataSet_modified.csv')

df['FAF'] = df['FAF'].replace({
    0.0: 'Never', 
    1.0: '1 or 2 days', 
    2.0: '2 or 4 days', 
    3.0: '4 or 5 days'
})
plt.figure(figsize=(12,6))
sns.scatterplot(x='FAF', y='BMI', hue='NObeyesdad', data=df)
plt.title('Physical Activity (FAF) vs BMI')
plt.xlabel('Frequency of Physical Activity')
plt.ylabel('BMI')
plt.xticks(rotation=45)
plt.show()


df['TUE'] = df['TUE'].replace({
    0.0: '0 - 2 hours', 
    1.0: '3 - 5 hours', 
    2.0: 'More than 5 hours'
})
plt.figure(figsize=(8,6))
sns.scatterplot(x='TUE', y='BMI', hue='NObeyesdad', data=df)
plt.title('Technology Use (TUE) vs BMI')
plt.xlabel('Time Using Technology')
plt.ylabel('BMI')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10,6))
sns.boxplot(x='MTRANS', y='BMI', data=df)
plt.title('BMI Distribution by Mode of Transportation')
plt.xlabel('Mode of Transportation')
plt.ylabel('BMI')
plt.xticks(rotation=45)
plt.show()
