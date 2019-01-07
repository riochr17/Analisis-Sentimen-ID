from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np 
from keras.models import model_from_json
from tfidf import TFIDF
from random import shuffle


class Analiser:
	'''
	variable for data training input and desired output
	'''
	xdata = []
	ydata = []

	'''
	variable for object attributes class
	'''
	tfidf_data = None
	model_loaded = None

	def __init__(self, training_data='data/training_all_random.csv'):
		self.preproses(training_data)
		return None

	'''
	Cleaning twitter data for training set.
	Current working: 
	- split new line
	- shuffle sentences order
	- split each sencetence by semicolon for separating data and desired output
	- append each data and desired output to the relevant variable
	'''
	def preproses(self, filepath):
		f = open(filepath)

		# split new line
		sents = f.read().split('\n')

		# shuffle all sentences order
		shuffle(sents)

		# on each sentence
		# - split by semicolon
		# - append to variable
		for sent in sents:
			temp = sent.split(';')
			if len(temp) == 2:
				self.xdata.append(temp[0])
				self.ydata.append([int(temp[1])])

		# prepare tfidf feature
		self.tfidf_data = TFIDF([self.xdata, self.ydata])

	def save_model(self, model, filename='model'):
		self.model_loaded = model

		# START SAVING MODEL
		# save model and weight
		# - save model
		model_json = model.to_json()
		with open("model/" + filename + ".json", "w") as json_file:
		    json_file.write(model_json)

		# - save weight
		model.save_weights("model/" + filename + ".h5")
		print("Saved model to disk")
		# END SAVING MODEL

	def load_model(self, filename='model'):
		model = Sequential()

		# START LOADING MODEL
		# Loading model and weight from saved data
		# - load model
		json_file = open("model/" + filename + ".json", 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json)
		
		# - load weights
		model.load_weights("model/" + filename + ".h5")
		print("Loaded model from disk")
		# END LOADING MODEL

		self.model_loaded = model
		return model

	'''
	Train data to build weighted network
	'''
	def train(self, output_filename = 'model'):
		X = self.tfidf_data.getOnlyX()
		Y = self.ydata

		# initialize model
		model = Sequential()

		# i am using heuristic for network layer by following guides:
		# - use only 4 layer
		# - number of nodes on each layer:
		# - - first layer 	: eq to data features dimension length (max. 300)
		# - - second layer 	: eq to .39 of first layer (activated by tanh)
		# - - third layer 	: 5 (activated by tanh)
		# - - output layer 	: 1 (activated by sigmoid)
		input_data_dimension = len(X[0])
		input_data_dimension = 300 if input_data_dimension > 300 else input_data_dimension

		model.add(Dense(units=int(0.39 * input_data_dimension), activation='tanh', input_dim=input_data_dimension))
		model.add(Dense(units=5, activation='tanh'))
		model.add(Dense(units=1, activation='sigmoid'))


		# loss error using binary crossentropy with backpropagation sgd optimizer
		# try lower learning rate on big number of training data
		learning_rate 	= .01
		loss_error		= 'binary_crossentropy'
		batch_size		= 1
		epoch 			= 10

		sgd = SGD(lr=learning_rate)
		model.compile(loss=loss_error, optimizer=sgd)

		# start building network
		model.fit(np.array(X), np.array(Y), batch_size=batch_size, nb_epoch=epoch)

		# saving model
		self.save_model(model, output_filename)

	'''
	Optional void,
	you can skip this if you won't retrain any of your model
	'''
	def retrain(self, output_filename):
		X = self.tfidf_data.getOnlyX()
		Y = self.ydata
		
		# loading model
		model = self.load_model(output_filename)

		# loss error using binary crossentropy with backpropagation sgd optimizer
		# lower learning used due to weighted network
		learning_rate 	= .005
		loss_error		= 'binary_crossentropy'
		batch_size		= 1
		epoch 			= 3

		sgd = SGD(lr=learning_rate)
		model.compile(loss=loss_error, optimizer=sgd)
		model.fit(np.array(X), np.array(Y), batch_size=batch_size, nb_epoch=epoch)

		# saving model
		self.save_model(model, output_filename)


	def getBinaryResult(self, x):
		return "POSITIF" if x >= 0.5 else "NEGATIF"

	'''
	Testing a sentences using saved weighted network
	'''
	def testFromTrained(self, x):
		if self.model_loaded == None:
			print "No model found! Load/train your model first to make a test"
			exit(0)
		
		# model.compile(loss='binary_crossentropy', optimizer=sgd)
		return self.getBinaryResult(self.model_loaded.predict_proba(np.array(x)))
