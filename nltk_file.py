import nltk # natural language toolkit
# nltk.download("punkt") - downloads the puntk package from nltk which contians a pretrained tokenizer hence we can use the nltk.word_tokenize()
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence) # returns the tokenized string

def stem(word):
    return stemmer.stem(word.lower()) #stems the word after it has been converted to lower case

def bag_of_words():
    pass

