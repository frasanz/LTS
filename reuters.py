from keras.datasets import reuters
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# This function is to vectorize train_data
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i, sequence in enumerate (sequences):
        results[i,sequence] = 1.
    return results

#This function is to vectorize labels
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i , label in enumerate(labels):
        results[i,label]=1.
    return results

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
print("The length of train data is:")
print(len(train_data))
print("This is the train_data #10:")
print(train_data[10])

#Get the word index and reverse it
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key,value) in word_index.items()])

# Because 0,1 and 2 are reverse index for padding, start of sequence and unknown
decoded_newswire = ' '.join([reverse_word_index.get(i-3,'?') for  i in train_data[0]])

#We are goint to print the new[0]
print("This is the train_data #0:")
print(decoded_newswire)

#Have a look at the labels
print("There are labels from #0 to #10")
print(train_labels[0:11])


#The next step is to prepare the data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels  = to_one_hot(test_labels)

# The last two can be done directly through keras
# from keras.utils.np_utils import to_categorical
# one_hot_train_labels = to_categorical(train_labels)
# one_hot_test_labels = to_categorical(test_labels)


model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

#Fit the model
history = model.fit(partial_x_train,partial_y_train, epochs=20, batch_size=512, validation_data = (x_val,y_val))

print(history.history)
#Draw the model, Draw the loss
loss     = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss)+1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Draw the model, Draw the accuracy
plt.clf()
acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Re_train until epoch #9 (overfitting after it)
model2 = models.Sequential()
model2.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model2.add(layers.Dense(64, activation='relu'))
model2.add(layers.Dense(46, activation='softmax'))



model2.compile(optimizer = 'rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model2.fit(partial_x_train, partial_y_train,epochs=9, batch_size=512, validation_data=(x_val, y_val))

results = model2.evaluate(x_test, one_hot_test_labels)

# Print the results 
print("I'm going to print the results")
print(results)


