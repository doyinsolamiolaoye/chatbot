import json
from nltk_file import tokenize, stem, bag_of_words

with open("intents.json", "r") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w) # extends the array instead of creatng an array of arrays
        xy.append((w,tag))

ignore_words = ["?", "!", ",", "." ]

all_words = [stem(word.lower()) for word in all_words if word not in ignore_words] #applies the stem function on each word in all_words that is not a punctuation

all_words = sorted(set(all_words)) #sorts it and removes duplicate words

