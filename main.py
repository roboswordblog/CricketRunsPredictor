import pandas as pd
from sklearn import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

df = pd.read_csv("cricket.csv")
# clean the data

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 16) # Strike rate, BFI, total matches played 
        self.fc2 = nn.Linear(16, 16)
        self.out = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

torch.manual_seed(41)
model = Model()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100

for i in range(epochs):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(f"Epoch {i}, Loss: {100-(loss.item()/1000)}")

with torch.no_grad():
    test_pred = model(X_test)
    test_loss = criterion(test_pred, y_test)
