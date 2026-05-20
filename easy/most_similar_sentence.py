from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read sentences from file
with open("sentence.txt", "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

# Take input
input_sentence = input("Enter input sentence: ")

# Add input to list
all_sentences = sentences + [input_sentence]

# Convert to TF-IDF
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(all_sentences)

# Compute similarity
similarity = cosine_similarity(tfidf)

# Get similarity scores of input with others
scores = similarity[-1][:-1]

# Find best match
best_index = scores.argmax()

# Output
print("\nMost similar sentence:")
print(sentences[best_index])

print("\nSimilarity score:")
print(scores[best_index])