import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.layer1 = nn.Linear(1, 32)
        self.layer2 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = self.layer2(x)
        return x
    
def ode_system(t, net):
    t = t.reshape(-1, 1)
    t = torch.tensor(t, dtype=torch.float32, requires_grad=True)
    t_0 = torch.zeros((1, 1))
    one = torch.ones((1, 1))

    u = net(t)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    ode_loss = u_t - torch.cos(2 * torch.pi * t)
    IC_loss = net(t_0) - one

    square_loss = ode_loss.pow(2) + IC_loss.pow(2)
    total_loss = torch.mean(square_loss)

    return total_loss

# 모델 및 옵티마이저 초기화
NN = PINN()
#optimizer = torch.optim.Adam(NN.parameters(), lr=0.001)
optimizer = torch.optim.LBFGS(NN.parameters(), lr=1.0, max_iter=6000, history_size=10)

train_t = (np.random.rand(30) * 2).reshape(-1, 1)
train_t = torch.tensor(train_t, dtype=torch.float32)
train_loss_record = []

# for itr in range(6000):
#     optimizer.zero_grad()
    
#     train_loss = ode_system(train_t, NN)
#     train_loss_record.append(train_loss.item())
    
#     train_loss.backward()
#     optimizer.step()
    
#     if itr % 1000 == 0:
#         print(train_loss.item())

# LBFGS 옵티마이저용 closure 함수 정의
def closure():
    optimizer.zero_grad()
    train_loss = ode_system(train_t, NN)
    train_loss.backward()
    train_loss_record.append(train_loss.item())
    return train_loss

# 학습 루프
for itr in range(6000):
    optimizer.step(closure)
    if itr % 1000 == 0:
        print(train_loss_record[-1])

plt.figure(figsize=(10, 8))
plt.plot(train_loss_record)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Over Iterations')
plt.show()


# Test 
test_t = np.linspace(0, 4, 100).reshape(-1, 1)
test_t_tensor = torch.tensor(test_t, dtype=torch.float32)

train_u = np.sin(2 * np.pi * train_t) / (2 * np.pi) + 1
true_u = np.sin(2 * np.pi * test_t) / (2 * np.pi) + 1

# PyTorch 모델의 예측 값 계산
NN.eval()  # 평가 모드로 변경
with torch.no_grad():
    pred_u_tensor = NN(test_t_tensor)

pred_u = pred_u_tensor.numpy().ravel()

# 그래프 그리기
plt.figure(figsize=(10, 8))
plt.plot(train_t, train_u, 'ok', label='Train')
plt.plot(test_t, true_u, '-k', label='True')
plt.plot(test_t, pred_u, '--r', label='Prediction')
plt.legend(fontsize=15)
plt.xlabel('t', fontsize=15)
plt.ylabel('u', fontsize=15)
plt.show()

