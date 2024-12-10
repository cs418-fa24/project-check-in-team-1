import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

file_path = 'ObesityDataSet.csv'

# Selecting categorical and numerical columns
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('NObeyesdad')  # Remove the target column from categorical columns list
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Creating a transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Define individual classifiers
rf = RandomForestClassifier(random_state=42)
svc = SVC(probability=True, random_state=42)
gb = GradientBoostingClassifier(random_state=42)

# Create the ensemble model using voting classifier
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('svc', svc),
        ('gb', gb)
    ],
    voting='soft'  # 'soft' uses the predicted probabilities to perform majority voting
)

# Pipeline with preprocessing and ensemble model
ensemble_pipeline = make_pipeline(preprocessor, ensemble_model)

# Splitting the data
X = data.drop('NObeyesdad', axis=1)
y = data['NObeyesdad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the ensemble model
ensemble_pipeline.fit(X_train, y_train)

# Predicting and evaluating the ensemble model
y_pred = ensemble_pipeline.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)

