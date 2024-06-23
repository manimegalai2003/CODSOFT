import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

def load_data(train_path, test_path, test_solution_path, description_path):
    train_data = pd.read_csv(train_path, delimiter=':::', header=None, names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])
    print("Training Data Loaded:")
    print(train_data.head())
    
    test_data = pd.read_csv(test_path, delimiter=':::', header=None, names=['ID', 'TITLE', 'DESCRIPTION'])
    print("\nTesting Data Loaded:")
    print(test_data.head())

    test_solutions = pd.read_csv(test_solution_path, delimiter=':::', header=None, names=['ID', 'TITLE', 'GENRE'])
    print("\nTest Data Solutions Loaded:")
    print(test_solutions.head())

    with open(description_path, 'r') as file:
        description = file.read()
    print("\nDescription File Loaded:")
    print(description[:500])

    return train_data, test_data, test_solutions, description

def preprocess_and_extract_features(train_data, test_data):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    
    X_train = vectorizer.fit_transform(train_data['DESCRIPTION'])
    X_test = vectorizer.transform(test_data['DESCRIPTION'])
    
    return X_train, X_test, vectorizer

def train_model(X_train, y_train):
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    return classifier

def predict_genres(classifier, X_test):
    y_pred = classifier.predict(X_test)
    return y_pred

# Define file paths
train_path = '/content/drive/MyDrive/CODSOFT/train_data.txt'
test_path = '/content/drive/MyDrive/CODSOFT/test_data.txt'
test_solution_path = '/content/drive/MyDrive/CODSOFT/test_data_solution.txt'
description_path = '/content/drive/MyDrive/CODSOFT/description.txt'

# Load the data
train_data, test_data, test_solutions, description = load_data(train_path, test_path, test_solution_path, description_path)

# Preprocess and extract features
X_train, X_test, vectorizer = preprocess_and_extract_features(train_data, test_data)

# Encode the target variable
y_train = train_data['GENRE']  # Assuming genre is in the 'GENRE' column

# Train the model
classifier = train_model(X_train, y_train)
print("Model Trained Successfully.")

# Predict genres for the test data
y_pred = predict_genres(classifier, X_test)

# Print the predicted genres
print("\nPredicted Genres:")
print(y_pred)

