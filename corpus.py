# Step 1: Start

# Step 2: Import required libraries
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Step 3: Read input sentence from user
sentence = input("Enter a sentence: ")

# Step 4: Tokenize sentence
tokens = word_tokenize(sentence)
print("\nTokens:")
print(tokens)

# Step 5: Convert all tokens to lowercase
tokens_lower = [word.lower() for word in tokens]

# Step 6: Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens_lower if word.isalpha() and word not in stop_words]

print("\nAfter Removing Stopwords:")
print(filtered_tokens)

# Step 7: Perform lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

print("\nAfter Lemmatization:")
print(lemmatized_tokens)

# Step 8: Apply POS tagging
pos_tags = pos_tag(lemmatized_tokens)

# Step 9: Display POS tagged words
print("\nPOS Tagged Words:")
for word, tag in pos_tags:
    print(word, " --> ", tag)

# Step 10: Stop