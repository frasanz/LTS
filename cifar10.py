#To develop my own keras of cifar10
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

cifar10 = tf.keras.datasets.cifar10

(train_images,train_labels), (test_images,test_labels) = cifar10.load_data()
train_images=train_images/255.0
print(train_labels)

#Show several pictures
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap = plt.cm.binary)
    plt.xlabel(train_labels[i])
#plt.show()

#Create the model
model  = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(32,32,3)),
    tf.keras.layers.Dense(512,activation=tf.nn.relu),
    tf.keras.layers.Dense(128,activation=tf.nn.relu),
    tf.keras.layers.Dense(128,activation=tf.nn.relu),
    tf.keras.layers.Dense(128,activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)])

#Compile the model
model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

history=model.fit(train_images,train_labels,epochs=5)
test_loss, test_acc = model.evaluate(test_images,test_labels)


