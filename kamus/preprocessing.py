"""#!/usr/bin/env python
import nltk
from nltk.tokenize import word_tokenize"""
#!/usr/bin/env python
import nltk
import re
from nltk.tokenize import word_tokenize
filename='tweet1clean.csv'
sfile='stopword.txt'
with open('tweet1clean.csv',"r") as f:
		data=f.read().split('\n')

def clean(text):
	t=[]
	for i in text: 
		 t.append(re.sub(r"http\S+", "", i))
	return t 
print clean(data)

