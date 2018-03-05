# Analisis Sentimen Twitter dengan TFIDF-ANN

Sentimen analisis twit dari Twitter untuk menentukan apakah sebuah twitter dianggap berbau POSITIF atau NEGATIF. Sentimen analisis ini menggunakan Multilayer Perceptron dengan ekstraksi fitur TF-IDF (Term Frequency and Inverse Document Frequency) 

## Installation

Pastikan scikit-learn, nltk, numpy, keras (dan backendnya), dan csv telah terpasang pada python.

## Usage

```python main.py```

## Testing

```python
# main.py
test = "rt @yusuflogen: ahok terkenal sbg gub tdk santun, tp kok kesan saya si ahok paling santun ya hari ini dbanding agus dan anis #debat2pilkada\u2026"
print test
print testFromTrained([td.transform(test)])

# output: POSITIF
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