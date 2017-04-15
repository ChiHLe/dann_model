from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout
from loading_amazon_dann_1 import load_amazon
from keras import backend as K
import numpy as np


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

    max_score = 0
for i in range(0,30):
    print("Trial # ", i)

    label_predictor = Sequential()
    label_predictor.add(Dense(50, activation='relu', input_shape=(max_words,)))
    label_predictor.add(Dropout(0.25))
    label_predictor.add(Dense(50, activation='relu'))
    label_predictor.add(Dropout(0.25))
    label_predictor.add(Dense(1, activation='sigmoid'))
    label_predictor.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    label_predictor.fit(X_train, y_train, validation_data=(X_test, y_test), shuffle=True,  epochs=3, batch_size=5,verbose=2)
    print("-------")
    # Final evaluation of the model
    scores=label_predictor.evaluate(X_new, y_new, verbose=2)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    if (scores[1]*100 > max_score):
        max_score = scores[1]*100
    print("Current maximum accuracy: %.2f%%" %max_score)

