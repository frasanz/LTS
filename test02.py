import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


array1 = np.array([10,80,21,9,15,50],dtype=float)
array2 = np.array([88,92,31,1,108,37],dtype=float)

l0    = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])
model.compile(loss='mean_squared_error', optimizer = tf.keras.optimizers.Adam(0.1))
history = model.fit(array1,array2,epochs=500)

plt.xlabel = ('Epoch number')
plt.ylabel = ('Loss magnitude')
plt.plot(history.history['loss'])

plt.show()


