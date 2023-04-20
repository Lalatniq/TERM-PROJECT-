Python 3.10.9 (tags/v3.10.9:1dd9be6, Dec  6 2022, 20:01:21) [MSC v.1934 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import tensorflow as tf
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import json
import random

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_words = ['?', '!']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # add documents in the corpus
        documents.append((w, intent['tag']))
        # add to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# sort classes
... classes = sorted(list(set(classes)))
... 
... # create training data
... training = []
... output_empty = [0] * len(classes)
... 
... for doc in documents:
...     bag = []
...     pattern_words = doc[0]
...     pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
...     for w in words:
...         bag.append(1) if w in pattern_words else bag.append(0)
... 
...     output_row = list(output_empty)
...     output_row[classes.index(doc[1])] = 1
... 
...     training.append([bag, output_row])
... 
... # shuffle the features and turn into np.array
... random.shuffle(training)
... training = np.array(training)
... 
... train_x = list(training[:, 0])
... train_y = list(training[:, 1])
... 
... # define the model
... model = tf.keras.Sequential([
...     tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
...     tf.keras.layers.Dropout(0.5),
...     tf.keras.layers.Dense(64, activation='relu'),
...     tf.keras.layers.Dropout(0.5),
...     tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
... ])
... 
... # compile model
... model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
... 
... # fit the model
history = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up

SyntaxError: multiple statements found while compiling a single statement
