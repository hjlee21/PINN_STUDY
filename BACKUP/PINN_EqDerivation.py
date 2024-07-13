import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
"""
def load_data_from_dir(dir_path, filename, col):
    data_list = []
    file_path = os.path.join(dir_path, filename)
    data = pd.read_csv(file_path)
    data_list.append(data[col])
    return data_list
"""

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.layer1 = nn.Linear(1, 64)
        self.layer2 = nn.Linear(64, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        return x

def loss_function(model, t, y, a, b):
    y_pred = model(t)
    mse_loss = nn.MSELoss()(y_pred, y)
    
    # 물리 기반 제약 조건: d2u/dx2 + b^2 * u = 0 (여기서 u = a * sin(b * x)로 가정)
    t.requires_grad_(True)
    y_pred = model(t)
    # dy_dx = torch.autograd.grad(y_pred, x, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
    # d2y_dx2 = torch.autograd.grad(dy_dx, x, grad_outputs=torch.ones_like(dy_dx), create_graph=True)[0]
    
    #physics_loss = torch.mean((a*np.sin + b**2 * y_pred) ** 2)
    
    # Physics
    
    dy_dt = y_pred - a * torch.sin(t)
    physics_loss = torch.mean(dy_dt ** 2)
    return mse_loss, physics_loss

# 데이터 생성
x_data = np.linspace(0, 4 * np.pi, 300).reshape(-1, 1)
y_data = 2 * np.sin(x_data)

x_train = x_data[0:200]
y_train = y_data[0:200]

x_tensor = torch.tensor(x_data[0:200], dtype=torch.float32)
x_test_tensor = torch.tensor(x_data, dtype=torch.float32)
y_tensor = torch.tensor(y_data[0:200], dtype=torch.float32)

# 모델 초기화
model = PINN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 상수 a와 b 초기화
a = torch.tensor(5.0, requires_grad=True)
b = torch.tensor(5.0, requires_grad=True)
optim_constants = optim.Adam([a], lr=0.001)


# 훈련
num_epochs = 10000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    optim_constants.zero_grad()
    mse_loss, physics_loss = loss_function(model, x_tensor, y_tensor, a, 0)
    
    log_mse = mse_loss.item()
    log_pl  = physics_loss.item()
    
    loss = mse_loss + physics_loss
    loss.backward()
    optimizer.step()
    optim_constants.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}, log_mse: {log_mse:.6f},log_pl: {log_pl:.6f}, a: {a.item():.4f}, b: {b.item():.4f}')

# 결과 시각화
model.eval()
with torch.no_grad():
    y_pred = model(x_test_tensor).numpy()

plt.plot(x_train, y_train, '*')
plt.plot(x_data, y_data, '--')
plt.plot(x_data, y_pred, label='Predicted')
plt.legend()
plt.show()