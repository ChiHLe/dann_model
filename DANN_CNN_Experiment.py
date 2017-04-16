from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv1D
from keras.layers import Flatten
from keras.layers import MaxPooling1D
from loading_amazon_dann_1 import load_amazon
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


if __name__ == '__main__':
    seed = 12342
    np.random.seed(seed)
    top_words = 5000
    data_folder = './data/'     # where the datasets are
    source_name = 'dvd'         # source domain: books, dvd, kitchen, or electronics
    target_name = 'electronics' # traget domain: books, dvd, kitchen, or electronics

    print("Loading data...")
    X_train, y_train, X_test, y_test, X_new, y_new, X_domain, y_domain = load_amazon(source_name, target_name, data_folder, verbose=True)
    max_words = X_train.shape[1]
    X_train = sequence.pad_sequences(X_train, maxlen=max_words)
    X_test = sequence.pad_sequences(X_test, maxlen=max_words)
    X_new = sequence.pad_sequences(X_new, maxlen=max_words)
    X_domain = sequence.pad_sequences(X_domain, maxlen=max_words)
    X_train = np.expand_dims(X_train, axis =2)
    X_test = np.expand_dims(X_test, axis=2)
    X_new = np.expand_dims(X_new, axis=2)
    X_domain = np.expand_dims(X_domain, axis=2)
    max_score = 0
    in_score = 0
for i in range(0,30):
    print("Trial # ", i)

    Model = Sequential()
    Model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(max_words,1)))
    Model.add(Dropout(0.25))
    Model.add(MaxPooling1D(pool_size=2))
    Model.add(Flatten())
    Model.add(Dense(50, activation='sigmoid'))
    Model.add(Dropout(0.25))
    Model.add(Dense(1, activation='sigmoid'))
    Model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    label_predictor = Sequential()
    label_predictor.add(Model)
    label_predictor.add(Dense(50, activation='relu'))
    label_predictor.add(Dropout(0.25))
    label_predictor.add(Dense(1, activation='sigmoid'))
    label_predictor.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    domain = Sequential()
    domain.add(Model)
    dense_layer = Dense(50, activation='relu')
    domain.add(GradientReversalLayer(1))
    domain.add(dense_layer)
    domain.add(Dropout(0.25))
    domain.add(Dense(1, activation='sigmoid'))
    domain.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    # Fit the model
    for epoch in range(0,101):
        print("Training labels")
        i = np.random.randint(0, X_train.shape[0]-32)
        label_predictor.train_on_batch(X_train[i:i+32], y_train[i:i+32])
        i = np.random.randint(0, X_domain.shape[0] - 32)
        print("Training domain")
        domain.train_on_batch(X_domain[i:i + 32], y_domain[i:i + 32])
        print("Testing")
        i = np.random.randint(0,X_test.shape[0]-100)
        current = label_predictor.test_on_batch(X_test[i:i+100], y_test[i:i+100])
        print("Current accuracy: ", current)
        if epoch % 5 == 0:
            scores = label_predictor.evaluate(X_test, y_test, verbose=1)
            print("In Domain Accuracy: %.2f%%" %(scores[1]*100))
            scores = label_predictor.evaluate(X_new, y_new, verbose=1)
            if (scores[1] * 100 > in_score):
               in_score = scores[1] * 100
            print("Label Accuracy: %.2f%%" % (scores[1] * 100))
            if (scores[1] * 100 > max_score):
               max_score = scores[1] * 100
            scores = label_predictor.evaluate(X_domain, y_domain, verbose=1)
            print("Domain Predictor Accuracy: %.2f%%" % (scores[1] * 100))
            print("Current target domain maximum accuracy: %.2f%%" % max_score)
            print("Current in domain maximum accuracy: %.2f%%" % in_score)
            transfer_loss = max_score - in_score
            print("Transfer Loss: %.2f%%" % transfer_loss)

    print("-------")
    # Final evaluation of the model


