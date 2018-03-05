from __future__ import division
import string
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from preproses import Preproses

class TFIDF:
	all_data = []
	onlyX	 = []

	# asumsi korpus terdiri dari array panjang 2, 
	# indeks 0 -> array xdata
	# indeks 1 -> array ydata
	# asumsi panjang indeks 0 dan indeks 1 sama
	def __init__(self, corpus):
		self.preprocessor = Preproses()
		self.tfidf_vectorizer = self.initVectorizer()
		self.tfidf_data = self.tfidf_vectorizer.fit_transform(corpus[0])
		self.fitData(corpus[1])

	def initVectorizer(self):
		tokenize = lambda sent: self.preprocessor.prep(sent)
		return TfidfVectorizer(norm='l2',
							   min_df=0, 
							   max_features=300,
							   use_idf=True, 
							   smooth_idf=False, 
							   sublinear_tf=True, 
							   tokenizer=tokenize)

	def fitData(self, ydata):
		i = 0
		self.all_data = []
		for count_0, doc in enumerate(self.tfidf_data.toarray()):
			self.onlyX.append(doc)
			self.all_data.append([doc, ydata[i]])
			i += 1

	def transform(self, sent):
		return self.tfidf_vectorizer.transform([sent]).toarray()[0]

	def getData(self):
		return self.all_data

	def getOnlyX(self):
		return self.onlyX

# def cosine_similarity(vector1, vector2):
# 	dot_product = sum(p*q for p,q in zip(vector1, vector2))
# 	magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
# 	if not magnitude:
# 		return 0

# 	return dot_product/magnitude
