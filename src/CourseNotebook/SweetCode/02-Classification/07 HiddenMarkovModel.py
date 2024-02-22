import numpy as np
from hmmlearn import hmm

# Define the states and observations
states = ["Rainy", "Sunny"]
observations = ["walk", "shop", "clean"]

# Create a dictionary to map states and observations to integers
state_map = {"Rainy": 0, "Sunny": 1}
obs_map = {"walk": 0, "shop": 1, "clean": 2}

# Define the HMM model parameters (transition and emission probabilities)
start_probability = np.array([0.6, 0.4])
transition_probability = np.array([[0.7, 0.3],
                                   [0.4, 0.6]])
emission_probability = np.array([[0.1, 0.4, 0.5],
                                 [0.6, 0.3, 0.1]])

# Initialize the HMM model
model = hmm.MultinomialHMM(n_components=len(states))

# Set the model parameters
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

# Define sample sequences for training
sequences = [
    ["walk", "shop", "clean"],
    ["walk", "clean", "clean"],
    ["walk", "walk", "shop"]
]

# Convert the sequences to numerical representation
X = [[obs_map[obs] for obs in seq] for seq in sequences]

# Fit the model to training data
model.fit(X)

# Predict the most likely sequence of hidden states for a given sequence of observations
observations = ["walk", "shop", "clean"]
obs_seq = [obs_map[obs] for obs in observations]
hidden_states = model.predict(np.array([obs_seq]))
predicted_states = [states[state] for state in hidden_states]
print("Predicted hidden states:", predicted_states)
