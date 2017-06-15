import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import csv

with open('train.csv', 'rb') as f:
	reader = csv.reader(f)
	train_x = list(reader)

labelfile = open("labels.txt", "r")
train_y = []
for line in labelfile:
	train_y.append(line.strip())
with open('test.csv', 'rb') as f:
	reader = csv.reader(f)
	test_x = list(reader)

labelsfile = open("labelstest.txt", "r")
test_y = []

for line in labelsfile:
	test_y.append(line.strip())

#add 1 as csv is 0 indexed
for j in range(len(train_x)):
	for i in range(len(train_x[j])):
		train_x[j][i] = int(train_x[j][i]) + 1

for i in range(len(test_x)):
	for j in range(len(test_x[i])):
		test_x[i][j] = int(test_x[i][j]) + 1

numpy.random.seed(7)
X = numpy.concatenate((train_x, test_y), axis=0)
num_words = len(numpy.unique(numpy.hstack(X)))

train_y = train_y[:205592]
test_y = test_y[:206385]

#word embedding
train_x = sequence.pad_sequences(train_x, maxlen=100)
test_x = sequence.pad_sequences(test_x, maxlen=100)

for i in range(len(train_x)):
	for j in range(len(train_x[i])):
		if train_x[i][j] <0 or train_x[i][j] >= num_words:
			train_x[i][j] = 0

for i in range(len(test_x)):
	for j in range(len(test_x[i])):
		if test_x[i][j] <0 or test_x[i][j] >= num_words:
			test_x[i][j] = 0
model = Sequential()
model.add(Embedding(num_words+1, 32))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(train_x, train_y,
          batch_size=32,
          epochs=15,
          validation_data=(test_x, test_y))
score, acc = model.evaluate(test_x, test_y,
                            batch_size=32)
print('score:', score)
print('accuracy:', acc)