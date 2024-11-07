import pandas as pd
import numpy as np

METER_TO_INCHES = 39.3701
KG_TO_LBS = 2.20462

df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

df['Height'] = df['Height'] * METER_TO_INCHES
df['Weight'] = df['Weight'] * KG_TO_LBS
df['Height'] = df['Height'].round(1)
df['Weight'] = df['Weight'].round(1)

def categorize_familyHistory(value):
    if value == 'yes':
        return '1'
    elif value == 'no':
        return '0'
df['family_history_with_overweight'] = df['family_history_with_overweight'].apply(categorize_familyHistory)

def categorize_FAVC(value):
    if value == 'yes':
        return '1'
    elif value == 'no':
        return '0'

df['FAVC'] = df['FAVC'].apply(categorize_familyHistory)

def categorize_caec(value):
    if value == 'no':
        return '0'
    elif value == 'Sometimes':
        return '1'
    elif value == 'Frequently':
        return '2'
    elif value == 'Always':
        return '3'

df['CAEC'] = df['CAEC'].apply(categorize_caec)

def categorize_SMOKE(value):
    if value == 'yes':
        return '1'
    elif value == 'no':
        return '0'

df['SMOKE'] = df['SMOKE'].apply(categorize_SMOKE)

def categorize_SCC(value):
    if value == 'yes':
        return '1'
    elif value == 'no':
        return '0'

df['SCC'] = df['SCC'].apply(categorize_SCC)

def categorize_CALC(value):
    if value == 'no':
        return '0'
    elif value == 'Sometimes':
        return '1'
    elif value == 'Frequently':
        return '2'
    elif value == 'Always':
        return '3'

df['CALC'] = df['CALC'].apply(categorize_CALC)

df['FCVC'] = df['FCVC'].round(0)
df['NCP'] = df['NCP'].round(0)
df['CH2O'] = df['CH2O'].round(0)
df['FAF'] = df['FAF'].round(0)
df['TUE'] = df['TUE'].round(0)
df['Age'] = np.floor(df['Age'])

df['BMI'] = (df['Weight'] / (df['Height'] ** 2)) * 703
df['BMI'] = df['BMI'].round(1)

df.to_csv('ObesityDataSet_modified.csv', index=False)










