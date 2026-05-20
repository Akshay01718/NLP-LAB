import nltk
from nltk.corpus import treebank
from nltk.tag import hmm

# nltk.download("treebank")

train_data = treebank.tagged_sents()

trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)

sentence = input("Enter a sentence: ")
tokens = sentence.split()
tagged_output = hmm_tagger.tag(tokens)

print("\nHMM POS tagged output:")
print(tagged_output)