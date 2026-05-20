import spacy
import benepar

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Add constituency parser
nlp.add_pipe("benepar", config={"model": "benepar_en3"})

sentence = input("Enter a sentence: ")

doc = nlp(sentence)

# ---------------- DEPENDENCY PARSE ----------------
print("\n--- Dependency Parse ---")
print("Word\tDependency\tHead")

for token in doc:
    print(f"{token.text}\t{token.dep_}\t\t{token.head.text}")

# ---------------- CONSTITUENCY PARSE ----------------
print("\n--- Constituency Parse ---")

for sent in doc.sents:
    print(sent._.parse_string)