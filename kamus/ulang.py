
def buka_file(filename):
	with open(filename,"r") as f:
			data=f.read().split('\n')
			return data 
def main(): 
	filename='tweet1clean.csv'
	sfile='stopword.txt'
	#running main 
	print buka_file(filename)

