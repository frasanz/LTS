''' Imports '''
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i,sequence]=1.
    return results


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print(train_data[0])
print(train_labels[0])
print(max([max(sequence) for sequence in train_data]))

#word_index is a dictionary (json) mapping words to an integer index
word_index = imdb.get_word_index()


#Reverse it
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


# # because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print(decoded_review)


## Vectorize inputs
x_train = vectorize_sequences(train_data)
x_test  = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test  = np.asarray(test_labels).astype('float32')

#create the model
model = models.Sequential()

model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#compile the model
model.compile(optimizer = optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy, metrics = [metrics.binary_accuracy])

# Get a slice
x_val = x_train[:10000]
partial_x_train = xtrain[10000:]

y_val = y_train[:10000]
partial_y_val = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs = 20, batch_size=512, validation_data = (x_val, y_val))

#Draw




