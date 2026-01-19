import math
import re
import pandas as pd
from collections import Counter

# -------------------------------
# Step 1: User Input
# -------------------------------
n = int(input("Enter number of documents: "))
documents = []

for i in range(n):
    doc = input(f"Enter document {i + 1}: ")
    documents.append(doc)

# -------------------------------
# Step 2: Preprocessing
# -------------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()

tokenized_docs = [preprocess(doc) for doc in documents]

# -------------------------------
# Step 3: Vocabulary
# -------------------------------
vocab = sorted(set(word for doc in tokenized_docs for word in doc))

# -------------------------------
# Step 4: Term Frequency (TF)
# -------------------------------
def compute_tf(doc_tokens):
    tf = {}
    total_words = len(doc_tokens)
    word_counts = Counter(doc_tokens)

    for word in vocab:
        tf[word] = word_counts[word] / total_words if total_words > 0 else 0

    return tf

tf_list = [compute_tf(doc) for doc in tokenized_docs]

# -------------------------------
# Step 5: Inverse Document Frequency (IDF)
# -------------------------------
def compute_idf(docs):
    idf = {}
    total_docs = len(docs)

    for word in vocab:
        doc_count = sum(1 for doc in docs if word in doc)
        idf[word] = math.log(total_docs / doc_count) if doc_count > 0 else 0

    return idf

idf = compute_idf(tokenized_docs)

# -------------------------------
# Step 6: TF-IDF Matrix
# -------------------------------
tfidf_matrix = []

for tf in tf_list:
    row = []
    for word in vocab:
        row.append(tf[word] * idf[word])
    tfidf_matrix.append(row)

# -------------------------------
# Step 7: Create Pandas DataFrame
# -------------------------------
df = pd.DataFrame(
    tfidf_matrix,
    columns=vocab,
    index=[f"Document {i+1}" for i in range(n)]
)

print("\nTF-IDF Matrix:\n")
print(df.round(4))
