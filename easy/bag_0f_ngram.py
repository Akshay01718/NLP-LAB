from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "I love machine learning",
    "Machine learning is fun",
    "I love AI"
]

# Create Bag of N-grams model
vectorizer = CountVectorizer(ngram_range=(2,2))

# Convert text into vectors
X = vectorizer.fit_transform(documents)

# Show vocabulary
print("Bigrams:")
print(vectorizer.get_feature_names_out())

# Show vectors
print("\nVectors:")
print(X.toarray())