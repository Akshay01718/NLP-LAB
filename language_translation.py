from translate import Translator

text = input("Enter text to translate: ")

lang = input("Enter target language (eg: ml, hi, fr, es): ")

translator = Translator(from_lang="en", to_lang=lang)
translation = translator.translate(text)

print("Original Text:", text)
print("Translated Text:", translation)