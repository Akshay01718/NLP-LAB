import nltk
nltk.download('gutenberg')
nltk.download('punkt')

from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
from collections import Counter

emma_text = gutenberg.raw('austen-emma.txt')
text = emma_text.lower()

words_token = word_tokenize(text)

total = len(words_token)
different = len(set(words_token))

word_count = Counter(words_token)

print("Total:", total)
print("Unique:", different)

percent = {}

for word in word_count.keys():
    percent[word] = (word_count[word] / total) * 100

word = input("Enter the word to count frequency: ")

if word in word_count.keys():
    print(f"Count of {word} = {word_count[word]}")
    print(f"% of {word} = {percent[word]}%")