# This is an example of one hot encoding using words
import numpy as np

samples = ('The cat sat on the mat.', 'The dog ate my homework.')

#First, build an index of all tokens in the data
token_index = {}
for sample in samples:
    # We simply tokenize the samples via the `split`method.
    # In real life, we would also strip puctuation and special characters
    # from the samples.
    for word in sample.split():
        if word not in token_index:
            # Assign a unique index to each unique word
            token_index[word] = len(token_index) + 1
            # Note the we don't attribute index 0 to anything

# Next, we vectorize our samples.
# We will only consider the first `max_length` words in each sample.
max_length = 10

# This is where we store our results:
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
for i, sampe in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i,j,index] = 1.

print(results)
