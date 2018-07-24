# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 10:34:14 2018

@author: AyushRastogi
"""
import pandas as pd
import os
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import json
import unicodedata
import sys
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
startTime = datetime.now()
from collections import Counter

os.path
os.getcwd() # Get the default working directory
path = r'C:\Users\AyushRastogi\OneDrive\LOS Files\Project 6 - Text Classification\Algorithm Approach\Data'
os.chdir(path)

data = pd.read_csv(path+r'\DataSegment0626.csv').astype(str)
#data = pd.read_csv(path+r'\DataSegment0710.csv').astype(str)
len(data.BOLType)
len(data.Annotation)

# -------------------------Dictionary Creation---------------
# Creating a dictionary with the two columns: BOLType and its BOLAnnotations
diction = dict([(i,[a]) for i, a in zip(data.BOLType, data.Annotation)])
diction
type(diction)

# ------------------------------------------------------------

# converting the dictionary into a .json (data-interchange format)
diction = json.dumps(diction)
loaded_r = json.loads(diction)
type(diction) # output should be str
type(loaded_r) # output should be dict

# a table structure to hold the different punctuation used
# sys.maxunicode: An integer giving the largest supported code point for a Unicode character
tbl = dict.fromkeys(i for i in range(sys.maxunicode) 
                    if unicodedata.category(chr(i)).startswith('P'))

# function to remove punctuations from sentences (unicode formatted strings)
def punc_remove(text):
    return text.translate(tbl)

# initialize the stemmer - obtain root of the words
stemmer = LancasterStemmer()

# variable to hold the Json data read from the file
data = None # by assigning none, variable does not hold any value at this time

# read the json file and load the training data
data = loaded_r
print(data)
type(data)

# get a list of all categories to train
categories = list(data.keys())
categories # a list with all keys
words = []

# a list of tuples with words in the sentence and category name
docs = []

# convert dictionaty key value combination to string
def keys2string(dictionary):
    """Recursive function which converts dictionary keys to strings"""
    if not isinstance(dictionary, dict):
        return dictionary
    return dict((str(k), keys2string(v)) 
        for k, v in dictionary.items())

keys2string(data)
data.keys()
data.values()

# Removing the stop words from the entire dataset
stop_words = set(stopwords.words('english'))

# NLTK - Natural Language Toolkit for Natural Language Processing
for each_category in data.keys():
    for each_sentence in data[each_category]:
        # remove any punctuation from the sentence
        each_sentence = punc_remove(each_sentence)
        # each_sentence = test_repl(each_sentence)
        print(each_sentence)
        # extract words from each sentence and append to the word list - Using NLTK Word Tokenize
        w = word_tokenize(each_sentence)
        w = [w for w in w if w not in stop_words]
        print("tokenized sentences: ", w)
        
        words.extend(w)
        docs.append((w, each_category))

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words]
# words = sorted(list(set(words))) # do not need to use this, since we need the frequency
print(words) # list
print(docs)  # list

# data has been converted to a dictionary
type(data) # dictionary

# https://docs.python.org/3/library/collections.html#collections.Counter
count = Counter(words)
count
dict(count) # convert to regular dictionary
most_com = count.most_common(20)
most_com # gives the most commonly used words along with the frequency - as a tuple

# here the original dictionary is data
# creating another dictionary d2 to save most common words
d2 = {}
for key, val in data.items():
    count = Counter(words)
    dict(count)
    d2[key] = count.most_common(20)

d2 # incorrect output
# convert the list 'words' into a dictionary with original keys