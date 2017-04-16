#!/usr/bin/env python
"""The training script for the DANN model."""

from __future__ import division
from __future__ import print_function

import csv
import os
import itertools
import sys

from collections import Counter
import cPickle as pkl

from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import GlobalMaxPool1D
from keras.layers import concatenate
from keras.models import Model
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

from sklearn.datasets import load_svmlight_files

BASE = os.path.dirname(os.path.realpath(__file__))


def yield_data(name):
    """Yields tokens from the dataset `name`."""
    fpath = os.path.join(BASE, 'data', name)
    if not os.path.exists(fpath):
        raise IOError('File not found: "%s"' % fpath)
    with open(fpath, 'rb') as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t')
        for sentiment, line in tsvin:
            yield int(sentiment) - 1, line.split(' ')


def yield_batches(name, sentence_length, batch_size, dictionary):
    """Yields batches of data."""
    arrs = []
    while True:
        for i, (sentiment, line) in enumerate(yield_data(name), 1):
            idxs = dictionary.convert_to_idx(line)
            idxs = idxs[:sentence_length] + [0] * (sentence_length - len(idxs))
            arrs.append((sentiment, idxs))
            if i % batch_size == 0:
                yield [np.asarray(a) for a in zip(*arrs)]
                arrs = []


class Dictionary(object):
    """Encodes the tokens as indices.
    Special indices:
        End-of-line: 0
        Out-of-vocab: 1
    """

    def __init__(self, location='dictionary.pkl', max_words=50000):
        self.location = os.path.join(BASE, 'data', location)
        self.max_words = max_words
        if not os.path.exists(self.location):
            self.train()
        with open(self.location, 'rb') as f:
            self.dictionary = pkl.load(f)

    def train(self):
        d = Counter()
        iterable = itertools.islice(yield_data('amzn_train.tsv'), 1000000)
        for i, (_, line) in enumerate(iterable, 1):
            d.update(line)
            if i % 1000 == 0:
                sys.stderr.write('\rprocessed %d lines' % i)
                sys.stderr.flush()
        sys.stderr.write('\rdone                \n')
        sys.stderr.flush()
        d = [i[0] for i in d.most_common(self.max_words)]
        d = dict((c, i) for i, c in enumerate(d, 2))
        with open(self.location, 'wb') as f:
            pkl.dump(d, f)

    @property
    def vocab_size(self):
        return len(self.dictionary)

    @property
    def rev_dict(self):
        if not hasattr(self, '_rev_dict') or not self._rev_dict:
            self._rev_dict = dict((i, c) for c, i in self.dictionary.items())
        return self._rev_dict

    def convert_to_idx(self, tokens):
        """Converts a list of tokens to a list of indices."""
        return [self.dictionary.get(t, 1) for t in tokens]

    def convert_to_tokens(self, indices):
        """Converts a list of indices to a list of tokens."""
        return [self.rev_dict.get(i, 'X') for i in indices]


def build_encoder(input_length, output_length, vocab_size):
    """Builds the encoder model."""
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


def build_adversary(input_length):
    """Predicts the domain of the vector."""
    input_layer = Input(shape=(input_length,))
    x = GradientReversalLayer(1)(input_layer)
    x = Dense(512, activation='relu')
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[input_layer], outputs=[x])
    return model


def build_training_models(input_length, latent_length, vocab_size):
    enc_model = build_encoder(input_length, latent_length, vocab_size)
    sent_model = build_sentiment(latent_length)
    adv_model = build_adversary(latent_length)

    input_layer = Input(shape=(input_length,))
    enc_out = enc_model(input_layer)
    sent_out = sent_model(enc_out)
    adv_out = adv_model(enc_out)
    train_model = Model(inputs=[input_layer], outputs=[adv_out])
    train_model.compile(optimizer='adam', loss='binary_crossentropy')
    adv_model.compile(optimizer='adam', loss='binary_crossentropy')

    sent_model = Model(inputs=[input_layer], outputs=[sent_out])
    sent_model.compile(optimizer='adam', loss='binary_crossentropy')

    return adv_model, train_model, sent_model, enc_model


if __name__ == '__main__':
    sentence_length = 100
    d = Dictionary()
    batch_size = 32
    test_size = 1000
    num_embed_dims = 512
    num_adv = 5

    num_epochs = 100
    num_batches = 100

    # Yields the training data.
    amzn_data = yield_batches('amzn_train.tsv', sentence_length, batch_size, d)
    yelp_data = yield_batches('yelp_train.tsv', sentence_length, batch_size, d)

    # Yields the testing data.
    amzn_test = yield_batches('amzn_test.tsv', sentence_length, test_size, d)
    yelp_test = yield_batches('yelp_test.tsv', sentence_length, test_size, d)

    # Builds the training models.
    adv_model, train_model, sent_model, enc_model = build_training_models(
        sentence_length, num_embed_dims, d.vocab_size)

    # Amazon -> 0, Yelp -> 1
    zeros, ones = np.zeros((batch_size,)), np.ones((batch_size,))
    zeros_test, ones_test = np.zeros((test_size,)), np.ones((test_size,))

    def print_eval(iterable, name, adv_target):
        sent, lines = iterable.next()
        preds = sent_model.predict([lines]).reshape(-1)
        adv_preds = adv_model.predict([lines]).reshape(-1)
        accuracy = np.mean(np.round(preds) == np.round(sent))
        adv_accuracy = np.mean(np.round(adv_preds) == np.round(adv_target))
        sys.stdout.write('   [ %s ] accuracy: %.3f  |  adv: %.3f\n'
                         % (name, accuracy, adv_accuracy))

    for epoch in range(1, num_epochs + 1):
        sys.stdout.write('\repoch %d                  \n' % epoch)
        print_eval(amzn_test, 'amazon', ones_test)
        print_eval(yelp_test, 'yelp', zeros_test)
        sys.stdout.flush()

        for batch_id in range(1, num_batches + 1):
            amzn_sent, amzn_lines = amzn_data.next()
            yelp_sent, yelp_lines = yelp_data.next()

            # Train the discriminator / adversary.
            for _ in range(num_adv):
                amzn_enc = enc_model.predict([amzn_lines])
                adv_model.train_on_batch([amzn_enc], [zeros])
                yelp_enc = enc_model.predict([yelp_lines])
                adv_model.train_on_batch([yelp_enc], [ones])

            # Trains the generator / sentiment analyzer.
            sent_model.train_on_batch([amzn_lines], [amzn_sent])
            # sent_model.train_on_batch([yelp_lines], [yelp_sent])

            # Trains the adversary part.
            #train_model.train_on_batch([amzn_lines], [ones])
            #train_model.train_on_batch([yelp_lines], [zeros])

            sys.stdout.write('\repoch %d, batch %d' % (epoch, batch_id))
            sys.stdout.flush()
