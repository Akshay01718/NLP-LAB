from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Read sentences from file
def read_sentences(filename):
    with open(filename, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    return sentences

# Step 2: Find most similar sentence
def find_most_similar(input_sentence, sentences):
    all_sentences = sentences + [input_sentence]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_sentences)

    similarity_matrix = cosine_similarity(tfidf_matrix)

    input_index = len(all_sentences) - 1
    similarities = similarity_matrix[input_index][:-1]

    max_index = similarities.argmax()
    max_score = similarities[max_index]

    return sentences[max_index], max_score, similarity_matrix

# Step 3: Main execution
if __name__ == "__main__":
    filename = "sentences.txt"
    input_sentence = input("Enter input sentence: ")

    sentences = read_sentences(filename)
    most_similar, score, sim_matrix = find_most_similar(input_sentence, sentences)

    print("\nMost similar sentence:")
    print(most_similar)

    print("\nSimilarity score:")
    print(score)

    print("\nSimilarity matrix:")
    print(sim_matrix)
