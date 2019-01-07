import re,string
import nltk
import csv
import numpy as np 
# from stemmer import MyStemmer

class Preproses:
	KATA_DASAR  = []
	DATA_KBBI 	= []

	def __init__(self):
		global KATA_DASAR
		global DATA_KBBI
		KATA_DASAR 	= [line.strip('\n')for line in open('kamus/rootword.txt')]
		DATA_KBBI	= [kamus.strip('\n').strip('\r') for kamus in open('kamus/kbba.txt')]

	def tokenize(self, tweet): 
		# this is very poor implementation of word tokenizer
		# try to rich tokenizer below using your best tools
		# token = nltk.word_tokenize(tweet)
		token = tweet.split(' ')
		return token

	def kbbi(self, token): 
		global DATA_KBBI

		#ubah list menjadi dictionary 
		dic={}
		for i in DATA_KBBI: 
			(key,val)=i.split('\t')
			dic[str(key)]=val

		#kbbi cocokan 
		final_string = ' '.join(str(dic.get(word, word)) for word in token).split()
		return final_string

	def normalize_token(self, _tokens):
		tokens = self.kbbi(_tokens)
		return tokens

	def preprocess(self, tweet):

		def hapus_tanda(tweet): 
			tanda_baca = set(string.punctuation)
			tweet = ''.join(ch for ch in tweet if ch not in tanda_baca)
			return tweet

		def hapus_katadouble(s): 
		    #look for 2 or more repetitions of character and replace with the character itself
		    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
		    return pattern.sub(r"\1\1", s)

		tweet=tweet.lower()
		tweet = re.sub(r'\\u\w\w\w\w', '', tweet)
		tweet=re.sub(r'http\S+','',tweet)
		#hapus @username
		tweet=re.sub('@[^\s]+','',tweet)
		#hapus #tagger 
		tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
		#hapus tanda baca
		tweet=hapus_tanda(tweet)
		#hapus angka dan angka yang berada dalam string 
		tweet=re.sub(r'\w*\d\w*', '',tweet).strip()
		#hapus repetisi karakter 
		tweet=hapus_katadouble(tweet)
		return tweet

	def prep(self, sent):
		return self.normalize_token(self.tokenize(self.preprocess(sent)))



