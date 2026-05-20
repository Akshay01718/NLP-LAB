import spacy
from nltk import Tree

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Convert SpaCy tree → NLTK Tree
def convert_tree(token):
    children = list(token.children)
    
    if children:
        return Tree(token.text, [convert_tree(child) for child in children])
    else:
        return token.text

# Main program
sentence = input("Enter a sentence: ").strip()

if sentence:
    doc = nlp(sentence)

    print("\n--- CONSTITUENCY PARSE ---")
    for sent in doc.sents:
        tree = convert_tree(sent.root)
        if isinstance(tree, Tree):
            tree.pretty_print()
        else:
            print(tree)

    print("\n--- DEPENDENCY PARSE ---")
    print("Word\tRelation\tHead")
    
    for token in doc:
        print(f"{token.text}\t{token.dep_}\t{token.head.text}")