import tensorflow as tf
import numpy as np
from tensorflow.python.framework import dtypes


class  TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

         # Placeholders for input, output and dropout
        #Input to the embedding layer is a tuple whose length is equal to the batch size, i.e, number of sentences in a batch.
        #Each element of the tuple is an array of numbers, representing sentence in the form of indexes into vocabulary

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        print(self.input_y)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
#=======================================================================================================
#embedding layer

        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            #create the embedding matrix, by default is true which mean is a learning parameter which will be learnt during training
            #when we use pretrained word embeddings we set it to false, so Trainable = false
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                trainable=False,
                name="W")
        #This statement takes as input, input_x from the previous step.
        #For each sentence, and for each word, it looks up (indexes) the embedding matrix
        #(w) and returns the corresponding embedding vector for each word of each sentence.
        #embedded_chars has a shape of [batch_size, sequence_length, embedding_vector_length]
        #here each element, example the first word of the first sentence is a real value
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
        #tensor is reshaped to [batch_size, sequence_length, embedding_vector_length, 1]
        #below in the expanded version embedded_chars_expanded, ku cdo word vector eshte size = 1 , instead of a real value 

            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

#output of the embedding layer Tensor of shape : [batch_size, sequence_length, embedding_vector_length, 1]

#=======================================================================================================
# Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):

 # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
#=======================================================================================================
# Apply nonlinearity
                #The bias needs to be added separately.
                #nonlinearity is being applied along with bias addition.
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
#=======================================================================================================
# Maxpooling over the outputs

                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
              
         # Combine all the pooled features 
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        # a reshape is performed to get a tensor of shape [batch_size, (3* 50)] = [batch_size , 150] = [215 , 150]
        # 
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
#=======================================================================================================
# Add dropout DROPOUT LAYER
#In this layer simply dropout is applied. Input and output shapes remain the same.
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

#=======================================================================================================
#Final (unnormalized) scores and predictions OUTPUT LAYER 
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            print(self.predictions)
#=======================================================================================================
# CalculateMean cross-entropy loss
#The loss is a measurement of the error our network makes, and our goal is to minimize it. 
#The standard loss function for categorization problems it the cross-entropy loss.
#Here, tf.nn.softmax_cross_entropy_with_logits is a convenience function that calculates 
#the cross-entropy loss for each class, given our scores and the correct input labels. 
#We then take the mean of the losses. We could also use the sum, but that makes it harder 
#to compare the loss across different batch sizes and train/dev data.
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

#=======================================================================================================
# Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


# reference from https://www.tensorflow.org/api_docs/python/tf/confusion_matrix 
#confusion matrix 
        with tf.name_scope("confusionmatrix"):
            confusion_matrix = tf.confusion_matrix(tf.argmax(self.input_y, 1),self.predictions, dtype=dtypes.int32,
                     name=None, weights=None)
            self.conf_matrix = confusion_matrix


