# Step 1: Start

# Step 2: Import required libraries
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree

# (Run once if not already downloaded)
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

# Step 3: Take input text from user
text = input("Enter a sentence: ")

# Step 4: Tokenize the input text
tokens = word_tokenize(text)

# Step 5: Assign POS tags
pos_tags = pos_tag(tokens)

# Step 6: Apply Named Entity Recognition
chunk_tree = ne_chunk(pos_tags)

# Step 7: Traverse the chunk tree to extract entities
print("\nNamed Entities:\n")

for subtree in chunk_tree:
    if isinstance(subtree, Tree):  # If subtree has a label (named entity)
        entity_name = " ".join([token for token, pos in subtree.leaves()])
        entity_type = subtree.label()
        print(f"Entity: {entity_name} | Type: {entity_type}")

# Step 9: Stop