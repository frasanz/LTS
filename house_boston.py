import numpy as np
from keras.datasets import boston_housing
from keras import models
from keras import layers
import matplotlib.pyplot as plt

# Load the data
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

#Normalize data
mean = train_data.mean(axis=0)
std  = train_data.std(axis=0)
train_data -= mean
train_data /=std
test_data -= mean
test_data /=std

#Function to build the model,
#Because we're going to use it a lot
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss = 'mse', metrics = ['mae'])
    return model


#K-fold validation
k=4
num_val_samples = len(train_data) // k
#We have to put this on 100
num_epochs = 5
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    # Prepare the validation data: data from partition k
    val_data    = train_data[i*num_val_samples: (i+1) * num_val_samples]
    val_targets = train_targets[i*num_val_samples: (i+1) * num_val_samples]

    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate(
            [ train_data[:i*num_val_samples], train_data[(i+1) * num_val_samples:]],
            axis=0)
    partial_train_targets = np.concatenate(
            [ train_targets[:i * num_val_samples],  train_targets[(i+1) * num_val_samples:]], axis=0)

    # Buld the Keras model (already compiled)
    model = build_model()

    # Train the model (in silent mode, verbose = 0)
    history = model.fit (partial_train_data,  partial_train_targets, 
            validation_data = (val_data, val_targets),
            epochs = num_epochs, batch_size = 1, verbose = 1)
    print(history.history)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]


#Plotting validation scores
plt.plot(range(1, len(average_mae_history)+1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()







