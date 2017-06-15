README
-------------------

Language
-------------------

Python 2.7.0 and above.

Libraries used
-------------------

fasttext
kerasscikit-learn
pandas
nltk

Fasttext
-------------------

Inside the fasttext.zip are train and test files of the yelp dataset as well as a python script to call fasttext classifier with parameters specified in the report.
Running the python script with call the classifier and train on yelp.train and test on yelp.test and print out the precision and recall scores

LSTM
-------------------
Inside LSTM.zip is 4 text files that were preprocessed using files used for HW1, to take the words in the dataset and index them, representing reviews as a list of numbers representing the index of a word. The labels are in the two remaining text files, 0 for negative and 1 for positive.

Python script reads in the data, and builds and compiles a keras model with an Embedding layer, LSTM layer with 0.2 dropout, and dense pooling layer.
Running LSTM on the 400,000 examples with the 4.5 million parameters takes just over 13 hours and gives 51% accuracy, with 90% training accuracy after 15 epochs.
The code it was based on used imdb dataset and is located here: https://github.com/fchollet/keras/blob/master/examples/imdb_lstm.py

It runs on 50,0000 examples in 90 minutes with 84% accuracy.

SVM
--------------------
Inside the logsvm.zip use train and test files of yelp dataset and svm.py file.  When running the svm.py file, following libraries are needed: scikit-learn, pandas and nltk. Running the file will give us accuracy, recall and precision.

Logistic Regression
--------------------

Inside logsvm.zip use train and test files of yelp dataset and lr.py file. For running the lr.py file, following libraries are needed: scikit-learn, pandas and nltk. After running, we obtain accuracy, precision and recall.


