import nltk # natural language toolkit
# nltk.download("punkt") - downloads the puntk package from nltk which contians a pretrained tokenizer hence we can use the nltk.word_tokenize()
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence) # returns the tokenized string

def stem(word):
    return stemmer.stem(word.lower()) #stems the word after it has been converted to lower case

def bag_of_words(tokenized_sentence, all_words): #takes in the tokenized sentence and checks if each token is in the all_words and represents with 1 if true and 0 if false
    """
    tokenized_sentence = ["hello","how","are","you"]
    all_words = ["hi","hello","I","you","bye","thank","cool"]
    bag =       [ 0 ,   1,     0,   0,    0,    0,      0   ]
    """
    tokenized_sentence = [stem(word) for word in tokenized_sentence] #stems each word in the toeknized sentence
    bag = np.zeros(len(all_words), dtype= np.float32) #initializes the bag of words
    for idx, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[idx] = 1.0

    return bag
    vfcdsfdfgfsdfsdsdsdssdsdsdsdfd