import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

df = pd.read_csv("cricket.csv")
df = df.drop(columns=[
    'Span', 'NO', 'HS',
    'SR', '100', '50', '0', '4s', '6s', "a"
])
for col in df.columns:
    if col != "Player":
        df[col] = pd.to_numeric(df[col], errors='coerce')
df['BFI'] = df['BF'] / df['Inns']
df = df.drop(["BF", "Inns"], axis=1)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16) # Strike rate, BFI, total matches played 
        self.fc2 = nn.Linear(16, 16)
        self.out = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

def getData(name):
    index = (df['Player'] == name).idxmax()
    return df.iloc[index].drop('Player').tolist()

def getRuns(name):
    index = (df['Player'] == name).idxmax()
    return df.iloc[index]["Runs"]
    
X = df.drop(columns=["Player", "Runs"]).values
y = df["Runs"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

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
