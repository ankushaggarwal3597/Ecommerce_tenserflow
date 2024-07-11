import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv('Train.csv')

# Drop ID column as it's not needed for modeling
df.drop(['ID'], axis=1, inplace=True)

# Data preprocessing
le = LabelEncoder()
cols = ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']
for col in cols:
    df[col] = le.fit_transform(df[col])

# Train test split
X = df.drop('Reached.on.Time_Y.N', axis=1)
y = df['Reached.on.Time_Y.N']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model building - Random Forest Classifier
rfc = RandomForestClassifier(random_state=0)
param_grid = {
    'max_depth': [4, 8, 12, 16],
    'min_samples_leaf': [2, 4, 6, 8],
    'min_samples_split': [2, 4, 6, 8],
    'criterion': ['gini', 'entropy']
}
grid_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_rfc.fit(X_train, y_train)
best_params_rfc = grid_rfc.best_params_
print('Best parameters for Random Forest Classifier:', best_params_rfc)

rfc_final = RandomForestClassifier(**best_params_rfc, random_state=0)
rfc_final.fit(X_train, y_train)
rfc_pred = rfc_final.predict(X_test)

# Model evaluation
print('Accuracy for Random Forest Classifier:', accuracy_score(y_test, rfc_pred))
print('Confusion Matrix for Random Forest Classifier:\n', confusion_matrix(y_test, rfc_pred))
print('Classification Report for Random Forest Classifier:\n', classification_report(y_test, rfc_pred))

# Save the model and scaler
with open('../web_app/models/rf_acc_68.pkl', 'wb') as model_file:
    pickle.dump(rfc_final, model_file)

with open('../web_app/models/normalizer.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
