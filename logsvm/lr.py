
import numpy as np
import pandas as pd
import re, nltk
import random
from nltk.stem.porter import PorterStemmer
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.linear_model import LogisticRegression       
import time

t1=time.time();

stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

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
 

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    stop_words = 'english',
    max_features = 85
)

Pos_Test = 286306
Neg_Test = 300096

#reading train and test data
test_file = 'Yelp_positive_test.csv'
train_file = 'Yelp_training_data.csv'


test_data = pd.read_csv(test_file, header=None)
test_data.columns = ["Text"]
train_data = pd.read_csv(train_file, header=None, delimiter=",").dropna()
train_data.columns = ["Sentiment","Text"] 
print(train_data.Text)

df = vectorizer.fit_transform(train_data.Text.tolist() + test_data.Text.tolist())
#converting feature vectore into an array
data_feature = df.toarray() 
vb = vectorizer.get_feature_names()

dist = np.sum(data_feature, axis=0)

#cross validation
X_train, X_test, y_train, y_test  = train_test_split(
        data_feature[0:len(train_data)], 
        train_data.Sentiment,
        train_size=0.80, 
        random_state=1234)


#defining the training model 
log_model = LogisticRegression(penalty = 'l1', tol = 0.004, C = 1000,max_iter = 300)
#fitting the model
log_model = log_model.fit(X=X_train, y=y_train)
#predicting on the test part for cross validation

y_pred = log_model.predict(X_test)
print 'Logisitic Cross-Validation'
print(classification_report(y_test, y_pred))

#defining parameters for testing

log_model = LogisticRegression(penalty = 'l1', tol = 0.004, C = 1000,max_iter = 300)

log_model = log_model.fit(X=data_feature[0:len(train_data)], y=train_data.Sentiment)

print("e")
#predicting on the test data
test_pred = log_model.predict(data_feature[len(train_data):])

spl = random.sample(xrange(len(test_pred)), Pos_Test)

print("f")
#Counting the no. of correct predictions for logistic regression model
PosLogcount = 0.0
for text, sentiment in zip(test_data.Text[spl], test_pred[spl]):
    if sentiment == 1:
        PosLogcount = PosLogcount + 1
######################################################
        
        test_file = 'Yelp_negative_test.csv'
train_file = 'Yelp_training_data.csv'


test_data = pd.read_csv(test_file, header=None)
test_data.columns = ["Text"]
train_data = pd.read_csv(train_file, header=None, delimiter=",").dropna()
train_data.columns = ["Sentiment","Text"]

df= vectorizer.fit_transform(train_data.Text.tolist() + test_data.Text.tolist())
data_feature= df.toarray()
vob = vectorizer.get_feature_names()

dist = np.sum(data_feature, axis=0)

X_train, X_test, y_train, y_test  = train_test_split(
        data_feature[0:len(train_data)], 
        train_data.Sentiment,
        train_size=0.80, 
        random_state=1234)

log_model = LogisticRegression(penalty = 'l2', tol = 0.003, C = 1000)#same parameter values as for positive
log_model = log_model.fit(X=X_train, y=y_train)

y_pred = log_model.predict(X_test)
print(classification_report(y_test, y_pred))

log_model = LogisticRegression(penalty = 'l1', tol = 0.004, C = 1000, max_iter = 300)
log_model = log_model.fit(X=data_feature[0:len(train_data)], y=train_data.Sentiment)

test_pred = log_model.predict(data_feature[len(train_data):])

# sample some of them
spl = random.sample(xrange(len(test_pred)), Neg_Test)
# print text and labels
NegLogcount = 0.0
for text, sentiment in zip(test_data.Text[spl], test_pred[spl]):
    if sentiment == 0:
        NegLogcount = NegLogcount + 1
    #print sentiment, text
        
accuracylog=(PosLogcount+NegLogcount)/float(Pos_Test+Neg_Test)
print 'Logistic Regression accuracy', accuracylog

F1_score_log = 2*(PosLogcount+NegLogcount)/float((2*(PosLogcount+NegLogcount))+(Pos_Test - PosLogcount)+(Neg_Test - NegLogcount))#F1 score calculation for Logistic Regression
print 'F1 score for Logistic Regression ', F1_score_log
t2=time.time();

total =t2-t1;
print("time required")
print(total)