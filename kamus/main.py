filename='tweet1clean.csv'
sfile='stopword.txt'
def buka_file(filename):
	with open(filename,"r") as f:
			data=f.read().split('\n')
			print data
