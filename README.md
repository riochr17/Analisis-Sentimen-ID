# Analisis Sentimen Twitter dengan TFIDF-ANN

Sentimen analisis twit dari Twitter untuk menentukan apakah sebuah twitter dianggap berbau POSITIF atau NEGATIF. Sentimen analisis ini menggunakan Multilayer Perceptron dengan ekstraksi fitur TF-IDF (Term Frequency and Inverse Document Frequency) 

## Installation

scikit-learn, nltk, numpy, keras (dan backendnya), csv

## Usage

```python
# analiser.py
# main class: Analiser(training_data)
# training_data default value = 'data/training_all_random.csv'

# 
# for main class example see main_*.py file, try run the file
#

# --
# main_pre_trained.py | load existing model
# analiser load training_data as base train data, load existing model

an = Analiser(training_data='data/training_all_random.csv')
an.load_model(filename='model')

# --
# main_training.py | train new model
# analiser load training_data as base train data, train the data, save the model

an = Analiser(training_data='data/training_all_random.csv')
an.train(filename='model')

# --
# main_re_training.py | retrain existing model
# analiser load training_data as base train data, load existing model, train the data, save the model

an = Analiser(training_data='data/training_all_random.csv')
an.retrain(filename='model')
```

## Testing

Pada data diatas, dataset yang digunakan adalah twit mengenai pilkada DKI Jakarta kemarin.

```python
# let analiser_instance is an instance of Analiser

test = "ahok itu pemimpin yang beres memimpin"
print test
print analiser_instance.testFromTrained([analiser_instance.tfidf_data.transform(test)])
# output: POSITIF

test = "ahok itu pemimpin yang ga beres memimpin"
print test
print analiser_instance.testFromTrained([analiser_instance.tfidf_data.transform(test)])
# output: NEGATIF
```

## Train / Re-train Network Forward/Backprop Parameters

```python
# analiser.py
# Change parameter for training
def train(self, output_filename = 'model'):
    ...
    learning_rate   = .01
    loss_error      = 'binary_crossentropy'
    batch_size      = 1
    epoch           = 10
    ...

# analiser.py
# Change parameter for retraining
def retrain(self, output_filename):
    ...
    learning_rate   = .005
    loss_error      = 'binary_crossentropy'
    batch_size      = 1
    epoch           = 3
    ...
```

## Contributing

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

## Credits

Dataset, koleksi barang-barang bahasa Indonesia, dan beberapa bagian preproses

https://github.com/ramaprakoso/analisis-sentimen

TF-IDF inspiration

https://appliedmachinelearning.wordpress.com/2017/02/12/sentiment-analysis-using-tf-idf-weighting-pythonscikit-learn/
