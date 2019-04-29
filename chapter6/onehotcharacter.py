#This is the one hot character encoding example
import string
import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate mt homework.']
characters = string.printable # All printable ASCII characters
token_index = dict(zip(range(1,len(characters)+1), characters))
print(characters)

max_length = 50
results = np.zeros((len(samples), max_length, max(token_index.keys()) + 1))
for i, sample in enumerate(samples):
    for j, character in enumerate (sample):
        index = token_index.get(character)
        results[i,j,index] = 1;

print results;
