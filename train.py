import json
import numpy as np
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
        w = tokenize(pattern) # returns each element/sentence in the pattern list as a token in a new list and assigns it to w 
        all_words.extend(w) # extends the list instead of creatng an array of arrays
        xy.append((w,tag)) # appends each toeknized list to a tag/label

ignore_words = ["?", "!", ",", "." ]

all_words = [stem(word.lower()) for word in all_words if word not in ignore_words] # applies the stem function on each word in all_words that is not a punctuation

all_words = sorted(set(all_words)) # sorts it and removes duplicate words
tags = sorted(set(tags))

X_train = [] # contains bag of words
y_train = [] # associate number for each tag

for pattern_sentence, tag in xy: # each toeknized word and its tag 
    bag = bag_of_words(pattern_sentence, all_words) # returns a bag of word for each tokenizded sentence
    X_train.append(bag) # contains list of bag of words for each tokenized sentence

    label = tags.index(tag) # converts the tag of the tokenized_sentence to index numberes
    y_train.append(label) # appends the index of the tag to the y_label list

X_train = np.array(X_train)
y_train = np.array(y_train)

print(y_train)







