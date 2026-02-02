import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("data.csv")

X_text = data["text"]
y_labels = data["label"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_labels)

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(X_text)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_,
    zero_division=0
))

new_text = ["The service was excellent"]
new_vector = vectorizer.transform(new_text)
prediction = model.predict(new_vector)

predicted_label = label_encoder.inverse_transform(prediction)
print("\nPrediction for new text:", predicted_label[0])
