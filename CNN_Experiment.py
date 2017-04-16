from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
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
    X_train = np.expand_dims(X_train, axis =2)
    X_test = np.expand_dims(X_test, axis=2)
    X_new = np.expand_dims(X_new, axis=2)
    in_score = 0
    max_score = 0
for i in range(0, 31):
    print("Trial # ", i)

    label_predictor = Sequential()
    label_predictor.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(max_words,1)))
    label_predictor.add(Dropout(0.25))
    label_predictor.add(MaxPooling1D(pool_size=2))
    label_predictor.add(Flatten())
    label_predictor.add(Dense(50, activation='sigmoid'))
    label_predictor.add(Dropout(0.25))
    label_predictor.add(Dense(1, activation='sigmoid'))
    label_predictor.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    label_predictor.fit(X_train, y_train, validation_data=(X_test, y_test), shuffle=True,  epochs=3, batch_size=5,verbose=2)
    print("-------")
    # Final evaluation of the model
    scores=label_predictor.evaluate(X_test, y_test, verbose=1)
    if (scores[1]*100 > in_score):
        in_score = scores[1]*100
    scores=label_predictor.evaluate(X_new, y_new, verbose=1)
    print("Current Target Accuracy: %.2f%%" % (scores[1]*100))
    if (scores[1]*100 > max_score):
        max_score = scores[1]*100
    print("Maximum Target domain accuracy: %.2f%%" %max_score)
    print("Maximum In domain accuracy: %.2f%%" % in_score)
    transfer_loss = max_score - in_score
    print("Transfer Loss: %.2f%%" % transfer_loss)
