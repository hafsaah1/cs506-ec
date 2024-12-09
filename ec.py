import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Load datasets
test_url = '/Users/auroraa/Downloads/cs-506-extra-credit/data/test.csv'
train_url = '/Users/auroraa/Downloads/cs-506-extra-credit/data/train.csv'

test = pd.read_csv(test_url)
train = pd.read_csv(train_url)

# Save test submission for later
test_submission = test.copy()

# Add helper columns
train['is_train'] = 1
test['is_train'] = 0

# Combine datasets for preprocessing
data = pd.concat([train, test], sort=False)

# Feature engineering
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date'] + ' ' + data['trans_time'])
data['dob'] = pd.to_datetime(data['dob'])
data['age'] = data['trans_date_trans_time'].dt.year - data['dob'].dt.year
data['trans_hour'] = data['trans_date_trans_time'].dt.hour

# Label encode categorical variables
le = LabelEncoder()
for col in ['gender', 'category']:
    data[col] = le.fit_transform(data[col])

# Frequency encoding for certain columns
for col in ['merchant', 'category', 'job', 'city', 'state', 'street']:
    freq_enc = data[col].value_counts().to_dict()
    data[col + '_freq_enc'] = data[col].map(freq_enc)

# Ensure 'amt' is float
data['amt'] = data['amt'].astype(float)

# Drop irrelevant columns
data.drop(
    ['trans_date_trans_time', 'trans_num', 'first', 'last', 'trans_time', 'dob', 'trans_date', 
     'merchant', 'job', 'city', 'state', 'street', 'zip', 'lat', 'long', 'merch_lat', 'merch_long'], 
    axis=1, 
    inplace=True, 
    errors='ignore'
)

# Split back into train and test sets
train = data[data['is_train'] == 1]
test = data[data['is_train'] == 0]

# Drop helper columns
train.drop(['is_train'], axis=1, inplace=True)
test.drop(['is_train', 'is_fraud'], axis=1, inplace=True)

# Separate features and target
X = train.drop('is_fraud', axis=1)
y = train['is_fraud']

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Scale the features
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)
test_scaled = scaler.transform(test)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_resampled_scaled, y_resampled, test_size=0.2, random_state=42
)

# Train XGBoost model
xgb = XGBClassifier(
    tree_method='hist',  # Use CPU-compatible method
    n_jobs=-1, 
    verbosity=2, 
    random_state=42,
    n_estimators=1000,
    max_depth=10,
    learning_rate=0.2,
    subsample=1.0,
    colsample_bytree=0.8,
    scale_pos_weight=len(y_resampled[y_resampled == 0]) / len(y_resampled[y_resampled == 1])
)

xgb.fit(X_train, y_train)

# Feature importance
importances = xgb.feature_importances_
feature_names = X.columns

# Plot feature importance
plt.figure(figsize=(10, 8))
plt.barh(feature_names, importances, color='skyblue')
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.title("Feature Importance from XGBoost")
plt.show()

# Retain top N important features
N = 10  # Retain top 10 features
important_features = feature_names[np.argsort(importances)[::-1][:N]]
X_train_important = X_train[:, np.argsort(importances)[::-1][:N]]
X_val_important = X_val[:, np.argsort(importances)[::-1][:N]]
test_scaled_important = test_scaled[:, np.argsort(importances)[::-1][:N]]

# Retrain the model with top features
xgb_important = XGBClassifier(
    tree_method='hist',
    n_jobs=-1,
    verbosity=2,
    random_state=42,
    n_estimators=1000,
    max_depth=10,
    learning_rate=0.2,
    subsample=1.0,
    colsample_bytree=0.8,
    scale_pos_weight=len(y_resampled[y_resampled == 0]) / len(y_resampled[y_resampled == 1])
)

xgb_important.fit(X_train_important, y_train)

# Evaluate the updated model
y_pred = xgb_important.predict(X_val_important)
y_pred_proba = xgb_important.predict_proba(X_val_important)[:, 1]

print("Classification Report:")
print(classification_report(y_val, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))
print("ROC AUC Score:", roc_auc_score(y_val, y_pred_proba))

# Predict on the test set
test_predictions = xgb_important.predict(test_scaled_important)

# Prepare the submission file
submission = pd.DataFrame({
    'id': test_submission['id'],
    'is_fraud': test_predictions.astype(int)
})

# Save the submission
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")
print("Top Features Used in the Updated Model:")
print(important_features)
