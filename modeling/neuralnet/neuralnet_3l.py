import torch
import torch.nn as nn
import torch.optim as optim


# ==================
# UNDER DEVELOPMENT
# ==================


class NN_3L(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NN_3L, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class NeuralNetwork_3L:
    def __init__(self, input_dim, output_dim=1, hidden_dim=32, lr=0.01, epochs=100):
        self.model = NN_3L(input_dim, hidden_dim, output_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs

    def train(self, X_train, y_train):
        self.model.train()
        for epoch in range(self.epochs):
            outputs = self.model(X_train)
            loss = self.criterion(outputs, y_train)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")

    def test(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_test)
            mse = self.criterion(preds, y_test)
            return preds, mse
