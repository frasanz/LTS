# This is to understand model from
# https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
model = Sequential()
model.add(Dense(2, input_dim=1, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
plot_model(model, show_shapes=True, show_layer_names=True)
