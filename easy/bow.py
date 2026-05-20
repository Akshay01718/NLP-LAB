from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "I love Natural Language Processing",
    "Natural Language Processing is fun fun",
    "I love learning NLP"
]

# Create Bag of Words model
vectorizer = CountVectorizer()

# Convert documents into BoW vectors
bow_matrix = vectorizer.fit_transform(documents)

# Vocabulary
print("Vocabulary:")
print(vectorizer.get_feature_names_out())

# BoW vectors
print("\nBag of Words Vectors:")
print(bow_matrix.toarray())

# Cosine similarity
similarity_matrix = cosine_similarity(bow_matrix)

print("\nCosine Similarity Matrix:")
print(similarity_matrix)