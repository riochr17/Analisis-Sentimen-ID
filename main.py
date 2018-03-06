from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np 
from keras.models import model_from_json
from tfidf import TFIDF
from random import shuffle

xdata = []
ydata = []

def preproses(filepath='data/training_all_random.csv'):
	global ydata
	global xdata

	f = open(filepath)
	sents = f.read().split('\n')
	shuffle(sents)
	for sent in sents:
		temp = sent.split(';')
		if len(temp) == 2:
			xdata.append(temp[0])
			ydata.append([int(temp[1])])

def train(X, Y):
	model = Sequential()

	indim = len(X[0])
	indim = 300 if indim > 300 else indim

	model.add(Dense(units=int(0.39 * indim), activation='tanh', input_dim=indim))
	model.add(Dense(units=5, activation='tanh'))
	model.add(Dense(units=1, activation='sigmoid'))

	sgd = SGD(lr=0.01)

	model.compile(loss='binary_crossentropy', optimizer=sgd)
	model.fit(np.array(X), np.array(Y), batch_size=1, nb_epoch=100)

	# serialize self.model to JSON
	model_json = model.to_json()
	with open("model/model.json", "w") as json_file:
	    json_file.write(model_json)

	# serialize weights to HDF5
	model.save_weights("model/model.h5")
	print("Saved model to disk")


def retrain_model(X, Y):
	model = Sequential()

	'''
	LOADED MODEL BEGIN
	'''
	# load json and create model
	json_file = open('model/model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	
	# load weights into new self.model
	model.load_weights("model/model.h5")
	print("Loaded model from disk")

	'''
	LOADED MODEL END
	'''

	sgd = SGD(lr=0.005)

	model.compile(loss='binary_crossentropy', optimizer=sgd)
	model.fit(np.array(X), np.array(Y), batch_size=1, nb_epoch=100)

	# serialize self.model to JSON
	model_json = model.to_json()
	with open("model/model.json", "w") as json_file:
	    json_file.write(model_json)

	# serialize weights to HDF5
	model.save_weights("model/model.h5")
	print("Saved model to disk")



def getBinaryResult(x):
	return "POSITIF" if x >= 0.5 else "NEGATIF"

def testFromTrained(x):
	model = Sequential()

	# load json and create model
	json_file = open('model/model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	
	# load weights into new self.model
	model.load_weights("model/model.h5")
	print("Loaded model from disk")

	sgd = SGD(lr=0.01)

	model.compile(loss='binary_crossentropy', optimizer=sgd)
	return getBinaryResult(model.predict_proba(np.array(x)))


preproses()
td = TFIDF([xdata, ydata])

# TRAINING
# train(td.getOnlyX(), ydata)

# RETRAINING
# retrain_model(td.getOnlyX(), ydata)

# TESTING
test = "ahok itu pemimpin yang beres memimpin"
print test
print testFromTrained([td.transform(test)])

test = "ahok itu pemimpin yang ga beres memimpin"
print test
print testFromTrained([td.transform(test)])




