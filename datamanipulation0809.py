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
df1['Annotation'][25] # Annotations for the first BOLType
type(df1['Annotation'][0]) # Annotations for each BOLId combined into a single str

# count the number of BOLs for each BOLType
print(df2['BOLType'].value_counts())

df_merged = pd.merge(df1, df2, on ='BOLId').sort_values("BOLType")
df_merged # Combined dataset with 3452 rows and 3 columns - BOLId, Annotation, BOLType
list (df_merged)
df_merged.dtypes
length = len(df_merged.index) # length of the colum: 3452

df_merged.to_csv(os.path.join(path,r'Data0713.csv')) # saving the file as .csv to the same folder 

df_merged['Freq'] = "" # creating a new column in df_merged with empty strings 

df_sample = df_merged

# ----- Data - Train and Test Split
length_data = len(df_merged.index)

df_all = df_merged.sample(n=length_data) # taking n random datapoints as a sample, if entire data reqd, use n = length_data 
df_randsample = df_merged.sample(n=1000)

df_all.to_csv(os.path.join(path,r'Dataset_Complete0809_unfiltered.csv'))
df_randsample.to_csv(os.path.join(path,r'Sample_Dataset_unfiltered.csv'))
df_sample = pd.read_csv(os.path.join(path,r'Sample_Dataset_unfiltered.csv'))
#df_sample = pd.read_csv(os.path.join(path,r'Sample0723_test.csv'))

print(df_sample['BOLType'].value_counts())
df_sample['Freq'] = ""

# ----- Text Manipulation ------

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
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

# print(list(df_sample['Freq'])) # to print the entire dataframe column

df_sample = df_sample.sort_values(by = 'BOLType') # sorting the dataframe by BOL Type
print(df_sample['BOLType']) # printing the list of BOL Types in the data

# ------------------------------------------------------

# convert dictionary to list 
# Convert the entire column 'Freq' to a LIST
import numpy as np
np.set_printoptions(suppress=True)
for i in range(len(df_sample)):
    df_sample['Freq'][i] = list(df_sample['Freq'][i].keys())
    # converting to set and back to list to remove non-unique elements from each BOL
    myset = set(df_sample['Freq'][i]) 
    df_sample['Freq'][i] = list(myset)

# -------------------------------------------------------   
# remove numbers and blank elements from string
from string import digits
remove_digits = str.maketrans('', '', digits)
for i in range(len(df_sample['Freq'])):
    df_sample['Freq'][i] = [item.translate(remove_digits) for item in df_sample['Freq'][i]]
    df_sample['Freq'][i] = list(filter(None, df_sample['Freq'][i]))

# removing the elements which are single or double character strings 
for i in range(len(df_sample['Freq'])):
    df_sample['Freq'][i] = [i for i in df_sample['Freq'][i] if len(i) > 2]
   
df_sample['Freq'][0]
# -------------------------------------------------------

# converting the list into a single string instead of multiple string elements
for i in range(len(df_sample['Freq'])):
    df_sample['Freq'][i] = ' '.join(df_sample['Freq'][i])
df_sample['Freq']

# Splitting dataset into Train and Test - after they have been filtered with NLTK
from sklearn.model_selection import train_test_split
train, test = train_test_split(df_sample, test_size = 0.25) # splitting the dataset into train = 75% and test = 25%

# getting the number of rows
df_sample.index 
train.index
test.index

# Exporting the training and test data into .csv files
train.to_csv(os.path.join(path,r'Sample_Dataset_Train.csv'))
test.to_csv(os.path.join(path,r'Sample_Dataset_Test.csv'))

# -------------------------------------------------------

# converting the category names to numbers 
#df_sample['BOLType'] = df_sample['BOLType'].astype('category') # converting the column to category
#df_sample['BOLCode'] = df_sample['BOLType'].cat.codes

# list(df_sample) # column names    
# merge list elements by BOLType
df_sample2 = train
# lists merged based on BOLType
df_sample2.index



df_sample2 = pd.read_csv(os.path.join(path,r'Sample_Dataset_Train.csv'))
df_sample2.info()
# store frequency of each BOLType as dictionary
counts = df_sample2['BOLType'].value_counts().to_dict()
#counts['CIG'] # access a specific key 

df_sample_test = df_sample2.groupby('BOLType', as_index=False).agg({'Freq': 'sum'})
print((df_sample_test['Freq']))
print(df_sample_test['BOLType'])

from collections import Counter
for i in range(len(df_sample_test['Freq'])):
    df_sample_test['Freq'][i] = Counter(df_sample_test['Freq'][i])

df_sample_test['Freq']

# 11 BOL Types

threshold = 0.70 # percent of occurence of string in each BOLType

for i in range(len(df_sample_test['Freq'])):
    if (df_sample_test['BOLType'][i] == 'CIG'):
        df_sample_test['Freq'][i] = {k: v /(counts['CIG'])  for k, v in df_sample_test['Freq'][i].items()}
        df_sample_test['Freq'][i] = { k:v for k, v in df_sample_test['Freq'][i].items() if v > threshold } # keep only the values which are above a threshold, using dict comprehension
        df_sample_test['Freq'][i] = sorted(df_sample_test['Freq'][i].items(), key=lambda kv: kv[1])
    if (df_sample_test['BOLType'][i] == 'FairmountSantrol'):
        df_sample_test['Freq'][i] = {k: v /(counts['FairmountSantrol'])  for k, v in df_sample_test['Freq'][i].items()}
        df_sample_test['Freq'][i] = { k:v for k, v in df_sample_test['Freq'][i].items() if v > threshold } # keep only the values which are above a threshold, using dict comprehension
        df_sample_test['Freq'][i] = sorted(df_sample_test['Freq'][i].items(), key=lambda kv: kv[1])
    if (df_sample_test['BOLType'][i] == 'Fila'):
        df_sample_test['Freq'][i] = {k: v /(counts['Fila'])  for k, v in df_sample_test['Freq'][i].items()}
        df_sample_test['Freq'][i] = { k:v for k, v in df_sample_test['Freq'][i].items() if v > threshold } # keep only the values which are above a threshold, using dict comprehension
        df_sample_test['Freq'][i] = sorted(df_sample_test['Freq'][i].items(), key=lambda kv: kv[1])
    if (df_sample_test['BOLType'][i] == 'HiCrush'):
        df_sample_test['Freq'][i] = {k: v /(counts['HiCrush'])  for k, v in df_sample_test['Freq'][i].items()}
        df_sample_test['Freq'][i] = { k:v for k, v in df_sample_test['Freq'][i].items() if v > threshold } # keep only the values which are above a threshold, using dict comprehension
        df_sample_test['Freq'][i] = sorted(df_sample_test['Freq'][i].items(), key=lambda kv: kv[1])
    if (df_sample_test['BOLType'][i] == 'MAALT'):
        df_sample_test['Freq'][i] = {k: v /(counts['MAALT'])  for k, v in df_sample_test['Freq'][i].items()}  
        df_sample_test['Freq'][i] = { k:v for k, v in df_sample_test['Freq'][i].items() if v > threshold } # keep only the values which are above a threshold, using dict comprehension
        df_sample_test['Freq'][i] = sorted(df_sample_test['Freq'][i].items(), key=lambda kv: kv[1])
    if (df_sample_test['BOLType'][i] == 'QuickSand'):
        df_sample_test['Freq'][i] = {k: v /(counts['QuickSand'])  for k, v in df_sample_test['Freq'][i].items()}
        df_sample_test['Freq'][i] = { k:v for k, v in df_sample_test['Freq'][i].items() if v > threshold } # keep only the values which are above a threshold, using dict comprehension
        df_sample_test['Freq'][i] = sorted(df_sample_test['Freq'][i].items(), key=lambda kv: kv[1])
    if (df_sample_test['BOLType'][i] == 'RFI'):
        df_sample_test['Freq'][i] = {k: v /(counts['RFI'])  for k, v in df_sample_test['Freq'][i].items()}  
        df_sample_test['Freq'][i] = { k:v for k, v in df_sample_test['Freq'][i].items() if v > threshold } # keep only the values which are above a threshold, using dict comprehension
        df_sample_test['Freq'][i] = sorted(df_sample_test['Freq'][i].items(), key=lambda kv: kv[1])
    if (df_sample_test['BOLType'][i] == 'Tidewater'):
        df_sample_test['Freq'][i] = {k: v /(counts['Tidewater'])  for k, v in df_sample_test['Freq'][i].items()}
        df_sample_test['Freq'][i] = { k:v for k, v in df_sample_test['Freq'][i].items() if v > threshold } # keep only the values which are above a threshold, using dict comprehension
        df_sample_test['Freq'][i] = sorted(df_sample_test['Freq'][i].items(), key=lambda kv: kv[1])
    if (df_sample_test['BOLType'][i] == 'TitanLansing'):
        df_sample_test['Freq'][i] = {k: v /(counts['TitanLansing'])  for k, v in df_sample_test['Freq'][i].items()}  
        df_sample_test['Freq'][i] = { k:v for k, v in df_sample_test['Freq'][i].items() if v > threshold } # keep only the values which are above a threshold, using dict comprehension
        df_sample_test['Freq'][i] = sorted(df_sample_test['Freq'][i].items(), key=lambda kv: kv[1])
    if (df_sample_test['BOLType'][i] == 'TwinEagle'):
        df_sample_test['Freq'][i] = {k: v /(counts['TwinEagle'])  for k, v in df_sample_test['Freq'][i].items()}
        df_sample_test['Freq'][i] = { k:v for k, v in df_sample_test['Freq'][i].items() if v > threshold } # keep only the values which are above a threshold, using dict comprehension
        df_sample_test['Freq'][i] = sorted(df_sample_test['Freq'][i].items(), key=lambda kv: kv[1])
    if (df_sample_test['BOLType'][i] == 'Unimin'):
        df_sample_test['Freq'][i] = {k: v /(counts['Unimin'])  for k, v in df_sample_test['Freq'][i].items()}  
        df_sample_test['Freq'][i] = { k:v for k, v in df_sample_test['Freq'][i].items() if v > threshold } # keep only the values which are above a threshold, using dict comprehension
        df_sample_test['Freq'][i] = sorted(df_sample_test['Freq'][i].items(), key=lambda kv: kv[1])

list(df_sample_test['Freq'])

# getting only the first element from each tuple i.e. getting rid of frequency/ratios
for i in range(len(df_sample_test['Freq'])):
    df_sample_test['Freq'][i] = [x[0] for x in df_sample_test['Freq'][i]]
print(df_sample_test['Freq'])

merged_list = np.concatenate(df_sample_test['Freq'].values)
unique, counts = np.unique(merged_list, return_counts=True)
master = unique[counts==1] # only include elements which occur once ie. count = 1
master

for i in range(len(df_sample_test['Freq'])):
    df_sample_test['Freq'][i] = np.intersect1d(df_sample_test['Freq'][i], master)
    print('\n')
    print("BOLType:", df_sample_test['BOLType'][i])
    print("BOL Identifier Keywords:", df_sample_test['Freq'][i] , '\n')

# ---------------------------------------------------------

# Machine Learning - Text Classification

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

labels = train['BOLType']
text = train['Annotation']

X_train, X_test, y_train, y_test = train_test_split(text, labels, random_state=0, test_size=0.25)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_transformed = tf_transformer.transform(X_train_counts)

X_test_counts = count_vect.transform(X_test)
X_test_transformed = tf_transformer.transform(X_test_counts)

labels = LabelEncoder()
y_train_labels_fit = labels.fit(y_train)
y_train_lables_trf = labels.transform(y_train)

print(labels.classes_)


from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

linear_svc = LinearSVC()
clf = linear_svc.fit(X_train_transformed,y_train_lables_trf)

calibrated_svc = CalibratedClassifierCV(base_estimator=linear_svc,
                                        cv="prefit")

calibrated_svc.fit(X_train_transformed,y_train_lables_trf)
predicted = calibrated_svc.predict(X_test_transformed)

df_results = pd.DataFrame(index=test.index, columns=labels.classes_)

for j in range(len(test.index)):
    i = test.index[j]
    to_predict = [test['Freq'][i]]
    p_count = count_vect.transform(to_predict)
    p_tfidf = tf_transformer.transform(p_count)
    print('Predicted probabilities of demo input string are')
    results_array = (calibrated_svc.predict_proba(p_tfidf)*100).round(3) # this is a numpy array
    print (results_array)
    max_value = np.max(results_array)
    maxpos = np.argmax(results_array)
    print('Max Probability:',max_value)
    print('Predicted BOL Type:', labels.classes_[maxpos], '\n')
    df_results.append(pd.DataFrame(calibrated_svc.predict_proba(p_tfidf)*100, columns=labels.classes_))

