import torch
import torch.nn as nn
import torch.optim as optim

# Parameters
input_size = 10  # size of each array
hidden_size = 32
num_classes = input_size

# Model
class PickMaxModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    def forward(self, x):
        return self.net(x)

model = PickMaxModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Generate dummy data
def generate_batch(batch_size=64):
    X = torch.rand(batch_size, input_size)
    y = torch.argmax(X, dim=1)  # index of the max value
    return X, y


# Training loop
for epoch in range(200):
    X, y = generate_batch()
    optimizer.zero_grad()
    out = model(X)
    loss = criterion(out, y)
    print(torch.argmax(out, dim=1))
    loss.backward()
    optimizer.step()

print("Training done.")