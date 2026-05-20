text = """Alice walked across the garden thinking about her adventures in Wonderland.
It was a warm afternoon, and Alice felt a gentle breeze pass by her.
As she moved past the trees, Alice noticed a small white rabbit watching her carefully.
The rabbit seemed nervous, but Alice simply smiled and continued her walk.
Later in the day, Alice sat under a large tree and wrote in her diary about everything she saw.
Although many strange things had happened before, Alice believed this day felt even more magical."""

stopwords = {'is', 'a', 'an', 'the', 'and', 'or', 'in', 'on', 'it', 'was', 'were', 'be', 'been'}

window = int(input("Enter window size: "))
target = input("Enter target word: ").lower()

# Clean text (remove punctuation + lowercase)
import string
tokens = text.lower().translate(str.maketrans('', '', string.punctuation)).split()

# Remove stopwords
tokens = [word for word in tokens if word not in stopwords]

print("\nConcordance:\n")

# Find concordance
for i, word in enumerate(tokens):
    if word == target:
        left = tokens[max(0, i - window):i]
        right = tokens[i + 1:i + window + 1]
        print(" ".join(left), word.upper(), " ".join(right))