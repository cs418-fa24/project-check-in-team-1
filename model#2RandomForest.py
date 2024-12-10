import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

file_path = 'ObesityDataSet.csv'
data = pd.read_csv(file_path)


categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('NObeyesdad') 
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])


rf_model = make_pipeline(
    preprocessor,
    RandomForestClassifier(random_state=42)
)


X = data.drop('NObeyesdad', axis=1)
y = data['NObeyesdad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)

