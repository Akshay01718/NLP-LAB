text = ("Alice walked across the garden thinking about her adventures in Wonderland. "
        "It was a warm afternoon, and Alice felt a gentle breeze pass by her. "
        "As she moved past the trees, Alice noticed a small white rabbit watching her carefully. "
        "The rabbit seemed nervous, but Alice simply smiled and continued her walk. "
        "Later in the day, Alice sat under a large tree and wrote in her diary about everything she saw. "
        "Although many strange things had happened before, Alice believed this day felt even more magical.")

stopwords = ['is', 'a', 'an', 'the', 'and', 'or', 'in', 'on', 'it', 'was', 'were', 'be', 'been']

window = int(input("Enter the window size: "))

punctuation = ['.', ',', '"', "'", '!', '?', ';', ':']

target = input("Enter the target word: ")

new_text = text.lower()

for p in punctuation:
    new_text = new_text.replace(p, "")

tokens = new_text.split()
filtered_tokens = []

for token in tokens:
    if token not in stopwords:
        filtered_tokens.append(token)

print("\nConcordance for the given word:\n")

for i, token in enumerate(filtered_tokens):
    if token == target:
        left_index = i - window
        if left_index < 0:
            left_index = 0

        left_context = filtered_tokens[left_index:i]

        right_index = i + window + 1
        right_context = filtered_tokens[i+1:right_index]

        line = " ".join(left_context) + " " + token.upper() + " " + " ".join(right_context)
        print(line)