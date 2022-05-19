import numpy as np
import nltk
#"punkt" is the package with pretrained tokeniser;; nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()



    # """
    # this function tokenise the sentences that is being fed as input in parameter
    # """
def tokenize(sentence):
    return nltk.word_tokenize(sentence)


    # """
    # stemming is a process of finding the root form of the word
    # """
def stem(word):
    #returns lower case stemmed word
    return stemmer.stem(word.lower())



    # """
    # returns a bag of words array, 1 for every word that's in that sentence otherwise it's a 0
    # """
def bag_of_words(tokenized_sentence, words):
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
