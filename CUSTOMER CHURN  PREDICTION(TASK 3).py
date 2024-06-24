import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
dataset = pd.read_csv('/content/drive/MyDrive/CODSOFT/Churn_Modelling.csv')

# Display information about the dataset
print("Dataset Description:")
print(dataset.describe())

print("\nDataset Head:")
print(dataset.head())

print("\nDataset Column Info:")
print(dataset.info())

# Drop unnecessary columns
dataset = dataset.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# One-hot encode categorical variables
dataset = pd.get_dummies(data=dataset, drop_first=True)

# Display unique values in the 'Gender' column if it exists
if 'Gender' in dataset.columns:
    print("\nUnique values in 'Gender' column:")
    print(dataset['Gender'].unique())

# Plot histogram of the 'Exited' column
plt.figure(figsize=(8, 6))
plt.hist(dataset['Exited'])
plt.xlabel('Exited')
plt.ylabel('Frequency')
plt.title('Histogram of Exited Column')
plt.show()

# Calculate the number of customers who exited
num_exited = (dataset['Exited'] == 1).sum()
print("\nNumber of customers who exited:", num_exited)

# Correlation with the 'Exited' column
corr_with_exited = dataset.corrwith(dataset['Exited'])
corr_with_exited.plot.bar(figsize=(10, 6), title='Correlation with Exited Column', rot=45)
plt.ylabel('Correlation')
plt.grid(True)
plt.show()

# Calculate correlation matrix
corr_matrix = dataset.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Prepare data for modeling
X = dataset.drop(columns='Exited')
y = dataset['Exited']

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Train and predict using Logistic Regression
clf_lr = LogisticRegression(random_state=0)
clf_lr.fit(X_train, y_train)
y_pred_lr = clf_lr.predict(X_test)

# Train and predict using Random Forest
clf_rf = RandomForestClassifier(random_state=0)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

# Train and predict using Gradient Boosting
clf_gb = GradientBoostingClassifier(random_state=0)
clf_gb.fit(X_train, y_train)
y_pred_gb = clf_gb.predict(X_test)

# Evaluate models
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    return pd.DataFrame([[name, acc, f1, prec, rec]], columns=['Model', 'Accuracy', 'F1', 'Precision', 'Recall'])

# Evaluate Logistic Regression
results_lr = evaluate_model('Logistic Regression', y_test, y_pred_lr)

# Evaluate Random Forest
results_rf = evaluate_model('Random Forest', y_test, y_pred_rf)

# Evaluate Gradient Boosting
results_gb = evaluate_model('Gradient Boosting', y_test, y_pred_gb)

# Combine results into a DataFrame
results = pd.concat([results_lr, results_rf, results_gb], ignore_index=True)
print("\nModel Evaluation Results:")
print(results)

# Confusion matrix for Logistic Regression
print("\nConfusion Matrix for Logistic Regression:")
print(confusion_matrix(y_test, y_pred_lr))

# Confusion matrix for Random Forest
print("\nConfusion Matrix for Random Forest:")
print(confusion_matrix(y_test, y_pred_rf))

# Confusion matrix for Gradient Boosting
print("\nConfusion Matrix for Gradient Boosting:")
print(confusion_matrix(y_test, y_pred_gb))
