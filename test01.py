# Test Celsius - fahrenheit
import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt

celsius_q    = np.array([-40,-10, 0, 8,15,22, 38],dtype=float)
fahrenheit_a = np.array([-40, 14,32,46,59,72,100],dtype=float) 
for i,c in enumerate(celsius_q):
    print("{} grados celsius = {} grados farenheit".format(c,fahrenheit_a[i]))

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])
model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=True)
print("Finished")
print("These are the layer variables: {}".format(l0.get_weights()))


m0 = tf.keras.layers.Dense(units=4, input_shape=[1])
m1 = tf.keras.layers.Dense(units=4)
m2 = tf.keras.layers.Dense(units=1)
model2 = tf.keras.Sequential([m0,m1,m2])
model2.compile(loss='mean_squared_error', optimizer = tf.keras.optimizers.Adam(0.1))
history2 = model2.fit(celsius_q,fahrenheit_a,epochs=500, verbose=True)
print("Finished model 2")
print(model.predict([100.0]))
print(model2.predict([100.0]))

n0 = tf.keras.layers.Dense(units=8,input_shape=[1])
n1 = tf.keras.layers.Dense(units=8)
n2 = tf.keras.layers.Dense(units=4)
n3 = tf.keras.layers.Dense(units=1)
model3 = tf.keras.Sequential([n0,n1,n2,n3])
model3.compile(loss='mean_squared_error', optimizer= tf.keras.optimizers.Adam(0.05))
history3 = model3.fit(celsius_q,fahrenheit_a,epochs=500, verbose=True)
print("Finished model 3")
print(model.predict([100.0]))
print(model2.predict([100.0]))
print(model3.predict([100.0]))

plt.xlabel = ('Epoch Number')
plt.ylabel = ('Loss Magnitude')
plt.plot(history.history['loss'],'r')
plt.plot(history2.history['loss'],'g')
plt.plot(history3.history['loss'],'b')

plt.show()


