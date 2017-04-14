from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, MaxPooling1D, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D
#from amazon_load import load_data
from loading_amazon_dann import load_amazon
from sklearn.datasets import load_svmlight_files
from sklearn import svm
from keras import backend as K
import numpy

def customized_loss(y_true, y_pred):
    return K.mean(1-K.square(y_pred-y_true), axis=-1)

if __name__ == '__main__':
    seed = 7
    numpy.random.seed(seed)
    top_words = 5000
    data_folder = './data/'     # where the datasets are
    source_name = 'dvd'         # source domain: books, dvd, kitchen, or electronics
    target_name = 'electronics' # traget domain: books, dvd, kitchen, or electronics




    print("Loading data...")
    X_train, y_train, X_test, y_test, X_new, y_new, X_domain, y_domain = load_amazon(source_name, target_name, data_folder, verbose=True)
    #(X_train, y_train), (X_test, y_test), (X_new, y_new)= load_data(num_words=top_words)
    max_words = X_train.shape[1]
    X_train = sequence.pad_sequences(X_train, maxlen=max_words)
    X_test = sequence.pad_sequences(X_test, maxlen=max_words)
    X_new = sequence.pad_sequences(X_new, maxlen=max_words)
    X_domain = sequence.pad_sequences(X_domain, maxlen=max_words)


    model = Sequential()
    model.add(Embedding(top_words, 32, input_length=max_words))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())


    label_predictor = Sequential()
    label_predictor.add(model)
    label_predictor.add(Dense(250, activation='relu'))
    label_predictor.add(Dense(1, activation='sigmoid'))
    label_predictor.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    domain_predictor = Sequential()
    domain_predictor.add(model)
    domain_predictor.add(Dense(250, activation='relu'))
    domain_predictor.add(Dense(1, activation='sigmoid'))
    domain_predictor.compile(loss=customized_loss, optimizer='adam', metrics=['accuracy'])

    # Fit the model
    for i in range(0,5):
        label_predictor.train_on_batch(X_train, y_train)
        domain_predictor.train_on_batch(X_domain, y_domain)
    # Final evaluation of the model
    scores = label_predictor.evaluate(X_new, y_new, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    #model_json = model.to_json()
    #with open("model.json","w") as json_file:
    #    json_file.write(model_json)
    #model.save_weights("model.h5")
    #print("Save model to disk")
    #model.save("model.h5")

    #json_file = open('model.json', 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()
    #loaded_model = model_from_json(loaded_model_json)
    #loaded_model = load_model("model.h5")
    #print("Loaded model from disk")
    #X_new_train = sequence.pad_sequences(X_new_train, maxlen=max_words)
    #X_new_test = sequence.pad_sequences(X_new_test, maxlen=max_words)

    #loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #loaded_model.fit(X_new_train, y_new_train, validation_data=(X_new_test, y_new_test), epochs=2, batch_size=128, verbose=2)
    #scores = model.evaluate(X_new_test, y_new_test, verbose=0)
    #print("Accuracy: %.2f%%" % (scores[1]*100))