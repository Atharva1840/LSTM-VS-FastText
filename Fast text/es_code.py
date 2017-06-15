import fasttext as ft
import numpy as np
import pandas as pd
import re, nltk

from nltk.stem.porter import PorterStemmer
# Fist download the dbpedia.train using https://github.com/facebookresearch/fastText/blob/master/classification-example.sh
# on test/ and move to the example directory

'''
This is prepocessing part
We can either use script or write the cocde
stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed
#Pre-processing of tweets


def tokenize(text): 
    text = re.sub("\d+", "", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = re.sub("@[^\s]+:/"," ",text)
    
    text = re.sub('[\s]+', ' ', text)
    text = re.sub("  ", " ", text)
    text = re.sub("haha","", text)
    text = re.sub("hahaha","", text)    
    text = text.strip('\'"?,.??')    
    text.replace('?',"")
    text.replace('??',"")    
    text = re.sub("`","", text)
    text = re.sub("#([^\s]+,$!&*)_.","", text)
    
    # tokenize
    tokens = nltk.word_tokenize(text)
    # stem
    stems = stem_tokens(tokens, stemmer)
    return stems
    '''
input_file = 'yelp.train'
output = 'class'
test_file = 'yelp.test'

# set params
dim=10
lr=0.1
epoch=5
min_count=1
word_ngrams=2
bucket=2000000
thread=4
silent=1
label_prefix='__label__'

# Train the classifier
classifier = ft.supervised(input_file, output, dim=dim, lr=lr, epoch=epoch,
    min_count=min_count, word_ngrams=word_ngrams, bucket=bucket,
    thread=thread, silent=silent, label_prefix=label_prefix)

# Test the classifier
result = classifier.test(test_file)
print 'P@1:', result.precision
print 'R@1:', result.recall
print 'Number of examples:', result.nexamples

