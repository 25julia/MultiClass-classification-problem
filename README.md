# Supervised Classification Problem

In this project we will be looking at an intent classification problem, basically a supervised classification problem using Convolutional Neural Networks with pre-trained word2vec vectors.

## Dataset

For this approach I used the open source benchmark of [Intento](https://github.com/snipsco/nlu-benchmark), which is a large dataset with 7 intents and 2000 queries per intent.

## Convolutional Neural Network

I was motivated by [Kim Yoon Paper](https://arxiv.org/abs/1408.5882) paper which is very famous for its performance in sentence classification task and by incorporating pretrained word vectors approach on top of convolutional neural networks, which makes the model perform better. I wanted to run and customize this approach for a multiclass classification problem

## Libraries

We will be using TensorFlow, sklearn to create the model, gensim to load the pretrained [Google vectors](https://code.google.com/archive/p/word2vec/) and of course python libraries like numpy, pandas etc to work with data.

## Problem statement 

The goal of this project is to create an intent classification model with pretrained word vectors on top of Convolutional Neural Networks, and to compare the results shown by [Snips](https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines) on different platforms like Wit, Luis, Api etc.


## Aproach Steps
 * Get and process data
  After downloading the open source data from this [repository](https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines), inside 2017-06-custom-intent-engines we will find the data that we need. There is a folder for each intent, and inside each folder there is a file `train_intent_full.json` which contains the raw data that we will use as input for our model. To process this data, I have created a script called extractQueries.js which concatenates the raw words into queries and saves them in a text file for each intent. Then we concatenate all data in each file in a file named allData.txt which holds all queries for each intent. I also generated a file called allLabels.txt which holds all labels for this queries.
  To shuffle the data I created a script which generates 3 folds, where we then use the 2 first folds as training data and the third fold as testing data. We can later run the model 3 times and then average the results, basically doing the 3 fold cross validation. For this purpuse I created a file called stratifiedFolds.js which generates stratifiedFolds.

 ***In the first round the training data will be the first and second folds,meanwhile the third fold will serve as testing set

 *  In the text file we define the layers of Convolutional Neural Networks, then in the train file we load the data with the help of functions defined in the data helper file. 
 Then for the training procedure defined in the train file, Tensor Flow session and graph concepts are used. The session is used to execute the graph operations, meanwhile the graph contains operations and tensors.




