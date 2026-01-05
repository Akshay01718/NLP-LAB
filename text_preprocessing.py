import re
import nltk
from io import StringIO
import sys
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK resources silently
old_stdout = sys.stdout
sys.stdout = StringIO()
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
finally:
    sys.stdout = old_stdout


def preprocess_text(text, use_lemmatization=True):
    print("Original Text:")
    print(text)

    # 1. Convert to lowercase
    text = text.lower()
    print("\nAfter Lowercase Conversion:")
    print(text)

    # 2. Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    print("\nAfter Removing Punctuation and Digits:")
    print(text)

    # 3. Tokenize
    tokens = word_tokenize(text)
    print("\nAfter Tokenization:")
    print(tokens)

    # 4. Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    print("\nAfter Stopword Removal:")
    print(tokens)

    # 5. Stemming or Lemmatization
    if use_lemmatization:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        print("\nAfter Lemmatization:")
        print(tokens)
    else:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        print("\nAfter Stemming:")
        print(tokens)

    return tokens


# Example usage
if __name__ == "__main__":
    sample_text = "Natural Language Processing (NLP) is AMAZING in 2025!!!"

    final_tokens = preprocess_text(sample_text)
    print("\nFinal Preprocessed Output:")
    print(final_tokens)
