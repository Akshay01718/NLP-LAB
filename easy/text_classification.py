import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load data
data = pd.read_csv("data.csv")

# Inputs and labels
X = data["text"]
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert text to numbers
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

# Accuracy
print("Accuracy:", (y_pred == y_test).mean())

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Test with new sentence
new_text = ["The service was very bad"]
new_vec = vectorizer.transform(new_text)

print("\nPrediction:", model.predict(new_vec)[0])