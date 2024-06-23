import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

# Load the datasets
train_df = pd.read_csv('/content/drive/MyDrive/CODSOFT/fraudTrain.csv')
test_df = pd.read_csv('/content/drive/MyDrive/CODSOFT/fraudTest.csv')

# Check data loading
print(train_df.head())
print(test_df.head())

# Convert 'amt' column to numeric
train_df['amt'] = pd.to_numeric(train_df['amt'], errors='coerce')
test_df['amt'] = pd.to_numeric(test_df['amt'], errors='coerce')

# Convert 'job' column to numeric (assuming it's binary with 'legitimate' and 'fraudulent' labels)
train_df['job'] = train_df['job'].map({'legitimate': 0, 'fraudulent': 1})
test_df['job'] = test_df['job'].map({'legitimate': 0, 'fraudulent': 1})

# Drop rows with NaN values after conversion
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# Split the data into features and target variable
X_train = train_df.drop(['job'], axis=1)
y_train = train_df['job']
X_test = test_df.drop(['job'], axis=1)
y_test = test_df['job']

# Check if X_train and X_test have samples
if len(X_train) == 0 or len(X_test) == 0:
    print("Error: X_train or X_test is empty.")
else:
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define and train the Logistic Regression model
    logreg = LogisticRegression()
    logreg.fit(X_train_scaled, y_train)

    # Make predictions and evaluate the model
    y_pred = logreg.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion)
    print("Classification Report:\n", report)
