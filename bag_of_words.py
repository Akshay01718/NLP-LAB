import re
from collections import Counter
import math

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize
    tokens = text.split()
    
    return tokens


def build_bag_of_words(documents):
    # Preprocess all documents
    processed_docs = [preprocess_text(doc) for doc in documents]
    
    # Build vocabulary
    vocabulary = sorted(set(word for doc in processed_docs for word in doc))
    
    # Create Bag of Words vectors
    bow_vectors = []
    for doc in processed_docs:
        word_count = Counter(doc)
        vector = [word_count[word] for word in vocabulary]
        bow_vectors.append(vector)
    
    return vocabulary, bow_vectors


def cosine_similarity(vec1, vec2):
    # Dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    # Magnitudes
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


# Example usage
if __name__ == "__main__":
    documents = [
        "I love Natural Language Processing",
        "Natural Language Processing is fun fun",
        "I love learning NLP"
    ]
    
    vocab, bow = build_bag_of_words(documents)
    
    print("Vocabulary:")
    print(vocab)
    
    print("\nBag of Words Vectors:")
    for i, vector in enumerate(bow):
        print(f"Document {i+1}:", vector)
    
    print("\nCosine Similarity Between Documents:")
    for i in range(len(bow)):
        for j in range(i + 1, len(bow)):
            similarity = cosine_similarity(bow[i], bow[j])
            print(f"Doc {i+1} vs Doc {j+1}: {similarity:.4f}")
