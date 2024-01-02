# -*- coding: utf-8 -*-
"""Deeplearning lab 1 st program.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lbn-zYaDer4XoACguEjwcWohJWzAOsCy

1a (random generation of weights)
"""

import numpy as np
from scipy.special import expit as activation_function
from scipy.stats import truncnorm

def truncated_normal(mean=0,sd=1,low=0,upp=10):
   return truncnorm( (low-mean)/sd,(upp-mean)/sd,loc=mean,scale=sd)

class Nnetwork:
  def __init__(self,no_of_in_nodes,no_of_out_nodes,no_of_hidden_nodes,learning_rate):
    self.no_of_in_nodes = no_of_in_nodes
    self.no_of_out_nodes = no_of_out_nodes
    self.no_of_hidden_nodes = no_of_hidden_nodes
    self.learning_rate = learning_rate
    self.create_weight_matrices()

  def create_weight_matrices(self):
    rad = 1 / np.sqrt(self.no_of_in_nodes)
    X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
    self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes,self.no_of_in_nodes))
    rad = 1 / np.sqrt(self.no_of_hidden_nodes)
    X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
    self.weights_hidden_out = X.rvs((self.no_of_out_nodes,self.no_of_hidden_nodes))



  def train(self, input_vector, target_vector):
    pass

  def run(self, input_vector):

    input_vector = np.array(input_vector, ndmin=2).T
    input_hidden = activation_function(self.weights_in_hidden @ input_vector)
    output_vector = activation_function(self.weights_hidden_out @ input_hidden)
    return output_vector

simple_network = Nnetwork(no_of_in_nodes=2,no_of_out_nodes=2,no_of_hidden_nodes=4,learning_rate=0.6)
simple_network.run([(3, 4)])

"""1b (keras)"""

from keras.models import Sequential
from keras.layers import Dense,Activation
import numpy as np

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])
model=Sequential()
model.add(Dense(2,input_shape=(2,)))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='mean_squared_error',optimizer='sgd',metrics=['accuracy'])
model.summary()

"""1c pythorch

"""

import torch
import torch.nn as nn
import torch.optim as optim

# Set random seed for reproducibility
torch.manual_seed(42)

# Generate synthetic data
X = torch.rand((100, 5))  # 100 samples, 5 features
y = (X.sum(dim=1) > 2.5).float().view(-1, 1)  # Binary classification task

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(5, 10)  # Input size: 5, Output size: 10
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)  # Output size: 1 (binary classification)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X)

    # Compute loss
    loss = criterion(outputs, y)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Make predictions on new data
new_data = torch.rand((5, 5))  # New data with 5 samples, 5 features
model.eval()
with torch.no_grad():
    predictions = (model(new_data) > 0).float()

print("Predictions:")
print(predictions)