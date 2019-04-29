from keras.datasets import imdb
from keras import preprocessing

# To create the embedding layer
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

# Number of words to be consider as features
max_features = 10000

# Cut tests after this number of words
# (among top max_features most common words) 
maxlen=20

# Load the data as list of integers.
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Now, this turns our list of integers
# into a 2D integer tensor of shape `(samples, maxlen)`
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test  = preprocessing.sequence.pad_sequences(x_test,  maxlen=maxlen)


model = Sequential()

# We specify the maximum input length to our Embedding layer
# so we can later flatten the embedded inputs
model.add(Embedding(10000, 8, input_length=maxlen))

# After the EMbedding layer,
# our activations have shape `(samples, maxlen, 8)`.

# We flatten the 3D tensor of embeddings
# into a 2D tensor of shape `(samples, maxlen * 8)`
model.add(Flatten())

# We add the classifier on top
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, 
                    y_train, epochs=10, batch_size=32, validation_split=0.2)


