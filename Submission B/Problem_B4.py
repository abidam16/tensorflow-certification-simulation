# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np


class customCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') is not None and logs.get('acc') > 0.91 and logs.get('val_acc') > 0.91:
            print('\nACC > 0.91\n')
            self.model.stop_training = True


def solution_B4():
    bbc = pd.read_csv(
        'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE
    # Using "shuffle=False"
    train, test = train_test_split(bbc, test_size=1-training_portion, shuffle=False)
    train_sentences = train['text']
    train_labels = train['category']
    validation_sentences = test['text']
    validation_labels = test['category']

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

    tokenizer.fit_on_texts(train_sentences)
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length, truncating=trunc_type)

    validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
    validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length, truncating=trunc_type)

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(train_labels)

    train_labels_sequences = np.array(label_tokenizer.texts_to_sequences(train_labels))
    validation_labels_sequences = np.array(label_tokenizer.texts_to_sequences(validation_labels))


    model = tf.keras.Sequential([
        # YOUR CODE HERE.
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    # Make sure you are using "sparse_categorical_crossentropy" as a loss fuction
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    callback = customCallback()
    model.fit(train_padded, train_labels_sequences,
              epochs=50,
              validation_data=(validation_padded, validation_labels_sequences),
              callbacks=[callback])

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    model.save("model_B4.h5")
