
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim



# ==================
# UNDER DEVELOPMENT
# ==================

class MHAttn_Regressor(nn.Module):

    def __init__(self, input_dim=60, d_model=72, num_heads=4, hidden_dim=64, dropout= 0.1):
        super(MHAttn_Regressor, self).__init__()
        
        # self.input_dim = input_dim
        # self.d_model = d_model        
        # if switch feature with samples: need to use synthetic functions and use their coeficients as scores
        self.embedding = nn.Linear(1, d_model)
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Output one regression value
        
    def forward(self, x):
        x = x.unsqueeze(-1) 
        x = self.embedding(x)                                     # (92, 42, 72)
        attn_output, attn_weight = self.attention(x, x, x)        # (92, 42, 72), (92, 42, 42)        
        x = self.activation(self.fc1(attn_output.mean(dim=1)))    # (92, 64)
        x = self.dropout(x)
        output = self.fc2(x)                                      # (92, 1)
        heatMap = attn_weight.mean(dim=0)                         # (42, 42)
        return output, heatMap
    
    def trainX(self, train_loader, criterion):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        num_epochs = 100
        for epoch in range(num_epochs):
            self.train()
            total_loss = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions, _ = self.forward(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

    def testX(self, test_loader, criterion):
        self.eval()
        with torch.no_grad():
            y_pred_list, y_true_list = [], []
            for batch_X, batch_y in test_loader:
                yp, heatMap = self.forward(batch_X)
                y_pred_list.append(yp)
                y_true_list.append(batch_y)

            y_pred = torch.cat(y_pred_list, dim=0)
            y_true = torch.cat(y_true_list, dim=0)
            test_loss = criterion(y_pred, y_true).item()
            print(f"Final Test Loss: {test_loss:.4f}")
            # y_org = scalerY.inverse_transform(y_true.numpy())
            # yp_org = scalerY.inverse_transform(y_pred.numpy())
            y_org = y_true.numpy()
            yp_org = y_pred.numpy()
            # print(f"mse: {mse(y_org, yp_org)}")
            # print(f"mae: {mae(y_org, yp_org)}")
            # print(f"smape: {smape(y_org, yp_org)}")

            heatMap_np = heatMap.numpy()
            med = np.max(heatMap_np) /2.0
            intercept_row = med * np.ones((1, heatMap_np.shape[1]))  
            intercept_col = med * np.ones((heatMap_np.shape[0] + 1, 1))
            
            heatMap_np = np.vstack((intercept_row, heatMap_np)) 
            heatMap_np = np.hstack((intercept_col, heatMap_np))

            plt.figure(figsize=(8, 8))
            plt.imshow(heatMap_np, cmap='viridis', aspect='auto')
            plt.colorbar(label="Attention Score")
            plt.xlabel("Key Index")
            plt.ylabel("Query Index")
            plt.title("Attention Score Heatmap")
            plt.savefig("attention_heatmap.png", dpi=300, bbox_inches='tight')
            # plt.show()
        return heatMap

