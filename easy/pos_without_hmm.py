import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download required resources (run once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Input sentence
text = "Virat Kohli plays cricket very well."

# Tokenize words
words = word_tokenize(text)

# POS tagging
tagged_words = pos_tag(words)

# Print result
print(tagged_words)