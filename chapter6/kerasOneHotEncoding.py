# This is only to change the title 

from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# We create a tokenizer, configured to only take into account
# the top-1000 most common on words
tokenizer = Tokenizer(num_words=1000)

# This build the word index
tokenizer.fit_on_texts(samples)

# This turns strings into lists of integer indices.
sequences = tokenizer.texts_to_sequences(samples)

# You can also directly get the one-hot binary represntations.
# Note that other vectorizetion modes than one-hot encoding are supported:
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

# Finally, this is how you can recover the word index that was computed
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

