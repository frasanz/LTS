# Simple representation of RNN

timesteps       = 100  # Numberof timesteps in the input sequence
input_features  =  32  # Dimensionality of the input feature space
output_features =  64  # Dimensionality of the output feature space

# This is our input data - just random noise for the sake of our example
input = np.random.random((timesteps, features))

# This is our "initial state": an all-zero vector:
state_t = np.zeros(output_features,))

# Create random weight matrices
W = np.random.random((input_features, output_features))
U = np.random.random((output_feeatures, output_features))
b = np.random.random((output_features,))

successive_outputs = []

for input_t in inputs:
