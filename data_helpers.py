import numpy as np
import re
import itertools
from collections import Counter
from sklearn import preprocessing
import tensorflow as tf


def clean_str(string):

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(corpus,labels):
    # Load data from files
    corpus_examples = list(open(corpus, "r", encoding="utf8").readlines())
    corpus_examples = [s.strip() for s in corpus_examples]
    x_text = corpus_examples
    x_text = [clean_str(sent) for sent in x_text]

    # Generate labels
    le = preprocessing.LabelEncoder()
    classes = list(open(labels, "r").readlines())
    classes = [s.strip() for s in classes]
    le.fit(classes) 
    label = le.transform(classes)
    a = np.array(label)
    y = np.zeros((13789, 7))
    y[np.arange(13789), a] = 1
    y
    return [x_text,y]


def batch_iter(data, batch_size, num_epochs, shuffle=False):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch, no need to shuffle in our case
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def load_word2vec_google(model,vocab_processor):
    # load embedding_vectors from the word2vec Google trained feature vectorss
    vocab_dict = vocab_processor._mapping
    sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])
    vocabulary = list(list(zip(*sorted_vocab))[0])
    #list of words in the google vocabulary
    google_vocab = model.vocab.keys()
    print(vocabulary)
    embedd = []
    for word in vocabulary:
         if word in google_vocab:
             if word != 'UNK':
                 embedd.append(model.wv[word])
         else:
             print("Word {} not in vocab".format(word))
             embedd.append([0] * 300)
    embedding = np.asarray(embedd, dtype="float32")
    return embedding








