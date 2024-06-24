import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import chardet  # Library for encoding detection
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Function to handle potential encoding errors while reading the CSV
def read_data(filename):
    try:
        # Try reading with UTF-8 encoding
        data = pd.read_csv(filename, encoding='utf-8')
        return data
    except UnicodeDecodeError:
        # If UTF-8 fails, attempt automatic encoding detection
        with open(filename, 'rb') as rawdata:
            result = chardet.detect(rawdata.read())
            encoding = result['encoding']
        # Read the data with the detected encoding
        data = pd.read_csv(filename, encoding=encoding)
        return data

# Load SMS dataset
data = read_data('/content/drive/MyDrive/CODSOFT/spam.csv')

# Print head, tail, info, and describe
print("Head of the dataset:")
print(data.head())

print("\nTail of the dataset:")
print(data.tail())

print("\nInfo of the dataset:")
print(data.info())

print("\nDescriptive statistics of the dataset:")
print(data.describe())

# Separate text messages and labels (spam or ham)
text = data['v2']
label = data['v1'].map({'ham': 0, 'spam': 1})  # Encode labels

# Preprocess text data (cleaning & converting to lowercase)
def clean_text(text):
    text = text.lower()  # convert to lowercase
    # Add additional cleaning steps like punctuation removal, stop word removal, etc.
    text = ''.join([char for char in text if char not in string.punctuation])
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    stemmer = PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

text = text.apply(clean_text)

# Feature Engineering - TF-IDF vectorization
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(text)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

# Train Multinomial Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Test model performance on unseen data
y_pred_nb = nb_model.predict(X_test)

# Train Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Test model performance on unseen data
y_pred_lr = lr_model.predict(X_test)

# Evaluate models (accuracy, precision, recall, etc.)
print("\nMultinomial Naive Bayes Model")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Classification Report:")
print(classification_report(y_test, y_pred_nb))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))

print("\nLogistic Regression Model")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Classification Report:")
print(classification_report(y_test, y_pred_lr))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

# Plot Confusion Matrix
def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

plot_confusion_matrix(y_test, y_pred_nb, "Multinomial Naive Bayes")
plot_confusion_matrix(y_test, y_pred_lr, "Logistic Regression")

# Function to predict spam for a new message
def predict_spam(message, model):
    # Clean and vectorize the message
    message = clean_text(message)
    message_features = vectorizer.transform([message])
    # Predict using the trained model
    prediction = model.predict(message_features)
    return prediction[0]

from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate precision, recall, f1-score, and accuracy for Multinomial Naive Bayes
precision_nb = precision_score(y_test, y_pred_nb)
recall_nb = recall_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb)
accuracy_nb = accuracy_score(y_test, y_pred_nb)

# Calculate precision, recall, f1-score, and accuracy for Logistic Regression
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

# Print evaluation metrics for Multinomial Naive Bayes
print("\nEvaluation metrics for Multinomial Naive Bayes Model:")
print("Precision:", precision_nb)
print("Recall:", recall_nb)
print("F1-score:", f1_nb)
print("Accuracy:", accuracy_nb)

# Print evaluation metrics for Logistic Regression
print("\nEvaluation metrics for Logistic Regression Model:")
print("Precision:", precision_lr)
print("Recall:", recall_lr)
print("F1-score:", f1_lr)
print("Accuracy:", accuracy_lr)


# Example usage
new_message = "XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap. xxxmobilemovieclub.com?n=QJKGIGHJJGCBL"
spam = predict_spam(new_message, nb_model)

print("\nPrediction for the new message:")
print("\nXXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap. xxxmobilemovieclub.com?n=QJKGIGHJJGCBL")

if spam:
    print("This message is likely spam.")
else:
    print("This message is likely not spam.")

new_message = "No no. I will check all rooms befor activities)"
spam = predict_spam(new_message, nb_model)

print("\nPrediction for the new message:")
print("\nNo no. I will check all rooms befor activities")

if spam:
    print("This message is likely spam.")
else:
    print("This message is likely not spam.")
