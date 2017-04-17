#!/usr/bin/env python
"""The training script for the DANN model."""

from __future__ import division
from __future__ import print_function

import csv
import os
import itertools
import sys

from collections import Counter
import pickle as pkl

from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import GlobalMaxPool1D
from keras.layers import concatenate
from keras.models import Model
from keras.datasets import imdb
from keras.engine.topology import Layer
import theano


import numpy as np
class ReverseGradient(theano.Op):
    """ theano operation to reverse the gradients
    Introduced in http://arxiv.org/pdf/1409.7495.pdf
    """

    view_map = {0: [0]}

    __props__ = ('hp_lambda', )

    def __init__(self, hp_lambda):
        super(ReverseGradient, self).__init__()
        self.hp_lambda = hp_lambda

    def make_node(self, x):
        assert hasattr(self, '_props'), "Your version of theano is too old to support __props__."
        x = theano.tensor.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        xin, = inputs
        xout, = output_storage
        xout[0] = xin

    def grad(self, input, output_gradients):
        return [-self.hp_lambda * output_gradients[0]]

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

class GradientReversalLayer(Layer):
    """ Reverse a gradient
    <feedforward> return input x
    <backward> return -lambda * delta
    """

    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversalLayer, self).__init__(**kwargs)
        self.hp_lambda = hp_lambda
        self.gr_op = ReverseGradient(self.hp_lambda)

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, mask=None):
        return self.gr_op(x)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"name": self.__class__.__name__,
                         "lambda": self.hp_lambda}
        base_config = super(GradientReversalLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



BASE = os.path.dirname(os.path.realpath(__file__))
word2index = imdb.get_word_index()


def yield_data(name):
    """Yields tokens from the dataset `name`."""
    if "yelp" in name:
        fpath = os.path.join(BASE, 'data', name)
        if not os.path.exists(fpath):
            raise IOError('File not found: "%s"' % fpath)
        with open(fpath, 'rb') as tsvin:
            tsvin = csv.reader(tsvin, delimiter='\t')
            for sentiment, line in tsvin:
                yield int(sentiment) - 1, line.split(' ')

def yield_batches(name, sentence_length, batch_size):
    """Yields batches of data."""
    arrs = []
    while True:
        for i, (sentiment, line) in enumerate(yield_data(name), 1):
            idxs = convert_to_idx(line)
            idxs = idxs[:sentence_length] + [0] * (sentence_length - len(idxs))
            arrs.append((sentiment, idxs))
            if i % batch_size == 0:
                yield [np.asarray(a) for a in zip(*arrs)]
                arrs = []

def convert_to_idx(line):
    return [word2index[token] for token in line if token in word2index and word2index[token] < 5000]
def build_encoder(input_length, output_length, vocab_size):
    """Builds the encoder model."""
    input_layer = Input(shape=(input_length,))
    input_layer = Input(shape=(input_length,))
    x = Embedding(vocab_size, 128)(input_layer)
    filters = [2, 3, 4]
    x = [Conv1D(64, length, padding='same')(x) for length in filters]
    x = concatenate(x, axis=-1)
    x = GlobalMaxPool1D()(x)
    x = Dense(output_length, activation='tanh')(x)
    model = Model(inputs=[input_layer], outputs=[x])
    return model


def build_sentiment(input_length):
    """Predicts if the vector corresponds to positive or negative."""
    input_layer = Input(shape=(input_length,))
    x = Dense(128, activation='tanh')(input_layer)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[input_layer], outputs=[x])
    return model


def build_training_models(input_length, latent_length, vocab_size):
    enc_model = build_encoder(input_length, latent_length, vocab_size)
    sent_model = build_sentiment(latent_length)


    input_layer = Input(shape=(input_length,))
    enc_out = enc_model(input_layer)
    sent_out = sent_model(enc_out)

    train_model = Model(inputs=[input_layer], outputs=[sent_out])
    train_model.compile(optimizer='adam', loss='binary_crossentropy')

    sent_model = Model(inputs=[input_layer], outputs=[sent_out])
    sent_model.compile(optimizer='adam', loss='binary_crossentropy')

    return train_model, sent_model, enc_model


if __name__ == '__main__':
    sentence_length = 100
    batch_size = 32
    test_size = 500
    num_embed_dims = 512
    num_adv = 5
    max_words = 5000

    num_epochs = 20
    num_batches = 100
    final_size = 25000
    # Yields the training data.

    train_index = 0
    test_index = 0
    (x_train, y_train), (x_test, y_test) = imdb.load_data(maxlen=sentence_length,num_words = max_words)

    yelp_data = yield_batches('yelp_train.tsv', sentence_length, batch_size)
    # Yields the testing data.

    yelp_test = yield_batches('yelp_test.tsv', sentence_length, test_size)


    # Yields final teting data
    yelp_final = yield_batches('yelp_test.tsv', sentence_length, final_size)


    # Builds the training models.
    train_model, sent_model, enc_model = build_training_models(sentence_length, num_embed_dims, max_words)

    # Amazon -> 0, Yelp -> 1

    def print_eval(iterable, name):
        sent, lines = iterable.next()
        preds = sent_model.predict([lines]).reshape(-1)
        accuracy = np.mean(np.round(preds) == np.round(sent))
        sys.stdout.write('   [ %s ] accuracy: %.3f \n' % (name, accuracy))

    def print_final(iterable, name):
        sent, lines = iterable.next()
        preds = sent_model.predict([lines]).reshape(-1)
        accuracy = np.mean(np.round(preds) == np.round(sent))
        sys.stdout.write('   [ %s ] accuracy: %.3f \n' % (name, accuracy))

    for epoch in range(1, num_epochs + 1):
        sys.stdout.write('\repoch %d                  \n' % epoch)

        sys.stdout.flush()

        if epoch == 20:
            print_eval(yelp_final, 'yelp')
            sent_model.evaluate(x_test, y_test, verbose=1)
        for batch_id in range(1, num_batches + 1):
            yelp_sent, yelp_lines = yelp_data.next()
            # Trains the generator / sentiment analyzer.
            sent_model.train_on_batch([yelp_lines], [yelp_sent])

            sys.stdout.write('\repoch %d, batch %d' % (epoch, batch_id))
            sys.stdout.flush()
