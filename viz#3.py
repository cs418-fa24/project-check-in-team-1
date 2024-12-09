import pandas as pd
import matplotlib.pyplot as plt


data_path = 'ObesityDataSet_modified.csv'
obesity_data = pd.read_csv(data_path)
grouped_smoking = obesity_data.groupby(['SMOKE', 'NObeyesdad', 'NCP']).size().unstack(fill_value=0).stack()
pivot_smoking = grouped_smoking.unstack(level=[0, 1]).fillna(0)
pivot_smoking.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='tab20')
plt.title('Impact of Smoking on Meal Frequency and Obesity Levels')
plt.xlabel('Meal Frequency (NCP)')
plt.ylabel('Count')
plt.legend(title='Smoking and Obesity Indicator', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


grouped_alcohol = obesity_data.groupby(['CALC', 'NObeyesdad', 'NCP']).size().unstack(fill_value=0).stack()
pivot_alcohol = grouped_alcohol.unstack(level=[0, 1]).fillna(0)
pivot_alcohol.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='tab20')
plt.title('Impact of Alcohol Consumption on Meal Frequency and Obesity Levels')
plt.xlabel('Meal Frequency (NCP)')
plt.ylabel('Count')
plt.legend(title='Alcohol Consumption and Obesity Indicator', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
