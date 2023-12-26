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
