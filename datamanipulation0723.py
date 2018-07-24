# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 15:07:25 2018

@author: AyushRastogi
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os
os.path

os.getcwd() # Get the default working directory
path = r'C:\Users\AyushRastogi\OneDrive\LOS Files\Project 6 - Text Classification\Algorithm Approach\Data'
os.chdir(path)

# Reading the first .csv file
df1_orig = pd.read_csv(r'C:\Users\AyushRastogi\OneDrive\LOS Files\Project 6 - Text Classification\Algorithm Approach\Data\dboAnnotation.csv')
# Reading the second .csv file
df2_orig = pd.read_csv(r'C:\Users\AyushRastogi\OneDrive\LOS Files\Project 6 - Text Classification\Algorithm Approach\Data\dboBOL.csv')

# getting column headers
list (df1_orig) 
list (df2_orig)

# selecting only the columns of interest
df1 = df1_orig[['BOLId', 'Annotation']].sort_values('BOLId') # sort by BOLId, ascending order
df2 = df2_orig[['BOLId', 'BOLType']].sort_values('BOLType') #sort by BOL Type, alphabetically

len(df1['BOLId'].unique()) # length of unique elements in a particular column
df2.head(10) # looking at first 10 values of the dataframe

# looking into the datatype of each column in the dataframe
df1.dtypes
df2.dtypes
type(df1['Annotation']) # alternate way to find datatype of a particular column, here dtype: object is for 'strings'

df1['Annotation'] = df1['Annotation'].astype(str) # Converting the column to string
df1['Annotation'] # also gives out the length of the column - 1,155,465

df2['BOLType'] = df2['BOLType'].astype(str) # Converting the column to string
df2['BOLType'] # also gives out the length of the column - 3452

# Use groupby/agg to aggregate the groups. For each group, apply set to find the unique strings, and ', '.join to concatenate the strings
#df1 = df1.groupby('BOLId').agg(lambda x: ', '.join(set(x))).reset_index()
df1 = df1.groupby('BOLId').agg(lambda x: ' '.join(x)).reset_index() # NON UNIQUE STRINGS

list (df1) # make sure both columns still present after the above operation
df1['Annotation'][0] # Annotations for the first BOLType
type(df1['Annotation'][0]) # Annotations for each BOLId combined into a single str

df2['BOLType'].value_counts() # count the number of BOLs for each BOLType

df_merged = pd.merge(df1, df2, on ='BOLId').sort_values("BOLType")
df_merged # Combined dataset with 3452 rows and 3 columns - BOLId, Annotation, BOLType
list (df_merged)
df_merged.dtypes
length = len(df_merged.index) # length of the colum: 3452

df_merged.to_csv(os.path.join(path,r'Data0713.csv')) # saving the file as .csv to the same folder 

df_merged['Freq'] = "" # creating a new column in df_merged with empty strings 


# ----- Working with Sample Data - Derived from the original datafile

df_randsample = df_merged.sample(n = 100) # taking 10 random datapoints as a sample 
df_randsample.to_csv(os.path.join(path,r'Sample0713.csv'))

df_sample = pd.read_csv(os.path.join(path,r'Sample0713.csv'))
df_sample['Freq'] = ""

range(len(df_sample))

# ----- Text Manipulation ------

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# import nltk
# nltk.download("stopwords")

# Removing the stop words from the entire dataset
stop_words = set(stopwords.words("english")) # list of stopwords 

# function to remove punctuations from sentences (unicode formatted strings)
# a table structure to hold the different punctuation used
# sys.maxunicode: An integer giving the largest supported code point for a Unicode character
from nltk.stem.lancaster import LancasterStemmer
import sys
import unicodedata
from collections import Counter

tbl = dict.fromkeys(i for i in range(sys.maxunicode) 
                    if unicodedata.category(chr(i)).startswith('P'))
def punc_remove(text):
    return text.translate(tbl)
## initialize the stemmer - obtain root of the words
stemmer = LancasterStemmer()

diction = {}

# trying manipulation on a single string out of the dataset
for i in range(len(df_sample)):
    df_sample['Freq'][i] = df_sample['Annotation'][i]
    df_sample['Freq'][i] = punc_remove(df_sample['Freq'][i]) # removing the punctuations
    word_tokens = word_tokenize(df_sample['Freq'][i]) # tokenizing the words
    #convert the text to lowercase so stopwords could be recognized
    lowercase = [item.lower() for item in word_tokens]
    filtered_words = [w for w in lowercase if not w in stop_words]
    
    string = ' '.join(filtered_words)
    df_sample['Freq'][i] = dict(Counter(string.split()).most_common())

df_sample['BOLType']
type(df_sample['Freq'][0]) #dict
type(df_sample['BOLType'][0]) #str
# ---------------------------------------------------------------------------------------

# Numpy Method 

# Convert the entire column 'Freq' to a list
import numpy as np
for i in range(len(df_sample)):
    df_sample['Freq'][i] = list(df_sample['Freq'][i].keys())

# list_test = list(df_sample['Freq'][0].keys())

# Convert the list to numpy array
#array = np.array(list_test, dtype='U10')
#array.dtype
#master = np.concatenate([])

freq = df_sample['BOLType'].value_counts().to_dict()
freq['HiCrush']


# method 1
df_sample[df_sample['BOLType'] == 'HiCrush']['Freq']
test1 = np.array([], dtype='U10')

for i in df_sample[df_sample['BOLType'] == 'HiCrush']['Freq']:
    test1 = np.concatenate([test1, i])

# method 2 for concatenating arrays
test2 = np.concatenate(df_sample[df_sample['BOLType'] == 'HiCrush']['Freq'].values)

unique, counts = np.unique(test1, return_counts=True)
dict1 = dict(zip(unique,(counts/freq['HiCrush'])))

unique, counts = np.unique(test2, return_counts=True)
dict2 = dict(zip(unique,counts/freq['HiCrush']))

dict1==dict2








freq = df_sample['BOLType'].value_counts().to_dict()
freq['TwinEagle']

for i in range(len(df_sample)):
    if (df_sample['BOLType'][i] =='CIG'):
        for m in range(freq['CIG']-1):
            list_CIG = []
            list_CIG = list(df_sample['Freq'][m].keys + df_sample['Freq'][m+1].keys)
        freq_key = list_CIG/(freq['CIG'])
    
    if (df_sample['Category'][i] =='Categ1'):
        for m in range(freq['Categ1']-1):
            list_Categ1 = []
            list_Categ1 = list(df_sample['Freq'][m].keys + df_sample['Freq'][m+1].keys)
        freq_key = list_Categ1/(freq['Categ1'])

# groupby and count instances of occurence of each unique word by using the func add_dict()


# using itertools and groupby - adding the values in the key:value combination 
from itertools import groupby, chain
a=[("13.5",100)]
b=[("14.5",100), ("15.5", 100)]
c=[("15.5",100), ("16.5", 100)]
input = sorted(chain(a,b,c), key=lambda x: x[0])
output = {}
for k, g in groupby(input, key=lambda x: x[0]):
  output[k] = sum(x[1] for x in g)
print (output)


# sort the BOLTypes alphabetically and add the keys as a list
for i in range(len(df_sample)-1):
    if (df_sample['BOLType'][i] == df_sample['BOLType'][i+1]):
        dict_comb = dict(list(df_sample['Freq'][i].items()) + list(df_sample['Freq'][i+1].items()))
    else:
        dict_comb = dict(list(df_sample['Freq'][i].items()))
    dict_comb

df_sample = df_sample.sort_values(by = ['BOLType'])

# Option 1
from itertools import chain
def add_dicts(s):
    #c = Counter()
    #s.apply(c.update)
    c = Counter(chain.from_iterable(e.keys() for e in s))
    return dict(c)   

df_sample = df_sample.groupby('BOLType').Freq.agg(add_dicts)
df_sample
# ------------------

# Option 2



# Iterate through the list of values and add them to a new dictionary by incrementing one by one
def func_iter(p):
    # start with an empty output dictionary
    out = {}
    # iterate through the keys in the dictionary
    for key in p:
       # iterate through the values in the sublist
       for val in p[key]:
          # check to see if we've seen this one before
          if not out.has_key(val):
             # if not, start it out at 0
             out[val] = 0
    
          # increment by one because we've seen it once more
          out[val] += 1
    print (out)


df_sample2 = df_sample.groupby('BOLType').Freq.agg(func_iter)
#---------------

class DictDiffer(object):
    """
    Calculate the difference between two dictionaries as:
    (1) items added
    (2) items removed
    (3) keys same in both but changed values
    (4) keys same in both and unchanged values
    """
    def __init__(self, current_dict, past_dict):
        self.current_dict, self.past_dict = current_dict, past_dict
        self.set_current, self.set_past = set(current_dict.keys()), set(past_dict.keys())
        self.intersect = self.set_current.intersection(self.set_past)
    def added(self):
        return self.set_current - self.intersect 
    def removed(self):
        return self.set_past - self.intersect 
    def changed(self):
        return set(o for o in self.intersect if self.past_dict[o] != self.current_dict[o])
    def unchanged(self):
        return set(o for o in self.intersect if self.past_dict[o] == self.current_dict[o])

d = DictDiffer(df_sample['Freq'][1], df_sample['Freq'][2])
print ("Changed:", d.changed())

# -----------------

# common keys in the entire list of dictionary
common_keys = set(df_sample['Freq'][1].keys())
for d in df_sample['Freq'][1:]:
    common_keys.intersection_update((d.keys()))
common_keys

def common_key(dict_list):
    common_keys = set(dict_list[1].keys())
    for d in dict_list[1:]:
        common_keys.intersection_update((d.keys()))
    common_keys
df_sample2 = df_sample.groupby('BOLType').Freq.agg(common_key)

df_sample['Freq'][0].keys()

## merge dictionary keys as a list 
#def merge_as_list(d):
#    newdict = {}
#    for i in range(len(df_sample)):
#        list(d.keys())
#df_sample4 = df_sample.groupby('BOLType').Freq.agg(merge_as_list)

## convert the dictionary keys into a list
#def conv_list(d):
#    for i in d.keys():
#        list()
#df_sample3 = df_sample.groupby('BOLType').Freq.agg(conv_list)
# --------------




'''
Options to try
1. Groupby by BOLType, see what effect it has on Freq - groupby.sum, groupby.agg, groupby with dict
2. Merge list of dictionaries by column name
3. Count number of instances of occurence of a key in a list of dictionaries
4. Drop unique keys (for each category), ie variable where freq=1, after merge all dicts by category ('inner')



for name, group in df_sample.groupby(['BOLType']):
    print(name)
    print(group)

#from collections import defaultdict
#dd = defaultdict(list)
#
#for d in (df_sample['Freq']): # you can list as many input dicts as you want here
#    for key, value in list(d.items()):
#        dd[key].append(value)

# delete unique keys (cases where value = 1 after groupby)
#for k, v in df_sample.items():
#    if v[0] = 1:
#        del df_sample[k]


# using itertools
import itertools
for k,v in itertools.groupby(df_sample['Freq'], key=lambda x:x['Categ1']):
    v = list(v)

df_sample['Freq']
for key in (df_sample['Freq']):
    if key in dic1: result.setdefault(key, []).append(dic1[key])

# groupby and count instances of occurence of each unique word by using the func add_dict()
def add_dicts(s):
    c = Counter()
    s.apply(c.update)
    return dict(c)   
df_sample = df_sample.groupby('BOLType').Freq.agg(add_dicts)

# write the dataframe to a csv file
df_sample.to_csv(os.path.join(path,r'Sample0713withFreq.csv'))

#merge dict based on condition
df_sample.groupby('BOLType').Freq.merge_dicts()
#

###
def print_keyval(df_sample):
    for key, value in list(df_sample.items()):
        print (key, value)
df_sample.groupby('BOLType').Freq.agg(print_keyval)
###


# count of categorical variables 'BOLType'
boltype_count = df_sample.groupby(['BOLType'])['Freq'].count()
type(boltype_count) # pandas series

# merge dictionary based on same BOLType, use zip 

{k: v for d in df_sample['Freq'] for k, v in df_sample.items()} # 
dict1_cond = {k:v for (k,v) in df_sample.items() if v <= 1}

# merge using dictionary comprehension
type(df_sample['Freq'])

#merge list of dictionary into a single list
#def merge(d):
#    {k: v for d in df_sample['Freq'] for k, v in d.items()}
#df_sample.groupby('BOLType').Freq.agg(merge)

from functools import reduce
reduce(lambda a, b: dict(a, **b),df_sample['Freq'])  # working


# https://codereview.stackexchange.com/questions/74462/merge-two-dict-in-separate-list-by-id?rq=1
# https://www.pythonforbeginners.com/dictionary-data-structure-in-python/dictionary-common-dictionary-operations/

# looping over keys
# for item in x.keys(): print item

# http://treyhunner.com/2016/02/how-to-merge-dictionaries-in-python/

# https://stackoverflow.com/questions/36950384/replacing-dictionary-values-as-a-ratio-of-total-values-in-the-dict

# fetch keys
b=[j[0] for i in df_sample for j in i.items()]
# print output
for k in list(set(b)):
    print ("{0}: {1}".format(k, b.count(k)))

'''

