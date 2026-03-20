import spacy
from nltk import Tree

nlp = spacy.load("en_core_web_sm")

def spacy_to_nltk(node):
    """
    Recursive function to convert SpaCy dependency nodes
    into an NLTK Tree structure.
    """
    if node.n_lefts + node.n_rights > 0:
        return Tree(f"{node.text} [{node.pos_}]",
                    [spacy_to_nltk(child) for child in node.children])
    else:
        return f"{node.text} [{node.pos_}]"


def main():
    sentence = input("Enter any sentence: ")
    if not sentence.strip():
        return

    doc = nlp(sentence)

    print("\n" + "=" * 40)
    print("CONSTITUENCY PARSING (Hierarchical)")
    print("=" * 40)

    for sent in doc.sents:
        nltk_tree = spacy_to_nltk(sent.root)
        if isinstance(nltk_tree, Tree):
            nltk_tree.pretty_print()
        else:
            print(f"Structure: {nltk_tree}")

    print("\n" + "=" * 40)
    print("DEPENDENCY PARSING (Relationships)")
    print("=" * 40)
    print(f"{'Word':<15} | {'Relation':<12} | {'Head (Parent)':<15}")
    print("-" * 45)

    for token in doc:
        print(f"{token.text:<15} | {token.dep_:<12} | {token.head.text:<15}")


if __name__ == "__main__":
    main()