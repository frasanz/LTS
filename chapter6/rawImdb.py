import os

# import keras and such things
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import numpy as np

# To draw models
from keras.utils.vis_utils import plot_model

imdb_dir  = './aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts  = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

# Tokenizing the tezt of the raw IMDB data
maxlen = 100                # We will cut reviews after 100 words
training_samples = 200      # We will be training on 200 samples
validation_samples = 10000  # We will be validating on 10000 samples
max_words = 10000           # We will only consider the top 10000 words of the dataset

tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' %len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of data tensor:' , data.shape)
print('Shape of label tensor:', labels.shape)

# Now, we are going to splir the data into a training set and a validation set
# But first, we are going to shuffle the data, since we started from data
# Where samples are ordered (all negative first, then all positive).
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = data[:training_samples]
x_val   = data[training_samples: training_samples+validation_samples]
y_val   = data[training_samples: training_samples+validation_samples]

# Parsing the glove word embeddings file
glove_dir = './glove.6B'
embeddings_index = {}

f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# Now, prepare the GloVe word embeddings matrix
embedding_dim    = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros
            embedding_matrix[i] = embedding_vector

# Now, the model 
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length = maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation = 'sigmoid'))

# Loading the matri of pre-trained word wmbeddings into the Embedding layer
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Training and evaludation
#model.compile(optimizer = 'rmsprop', loss='binary_crossentropy', metrics=['acc'])
#history = model.fit(x_train, y_train, 
#        epochs=10, batch_size=32, validation_data=(x_val, y_val))
#model.save_weights('pre_trained_glove_model.h5')

