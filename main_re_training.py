from analiser import Analiser

# start analiser with set training data
an = Analiser(training_data='data/training_all_random.csv')

# retrain existing model
an.retrain(filename='model')

test = "ahok itu pemimpin yang beres memimpin"
print test
print an.testFromTrained([an.tfidf_data.transform(test)])

test = "ahok itu pemimpin yang ga beres memimpin"
print test
print an.testFromTrained([an.tfidf_data.transform(test)])
