import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset

Training = True
Drawing = False
# 학습 가능한 변수 a와 b를 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

a = torch.nn.Parameter(torch.tensor(5.0, device=device)) # 초기값 설정
b = torch.nn.Parameter(torch.tensor(6.0, device=device)) # 초기값 설정
c = torch.nn.Parameter(torch.tensor(7.0, device=device)) # 초기값 설정
d = torch.nn.Parameter(torch.tensor(8.0, device=device)) # 초기값 설정

def generate_data_total(num_points=200):
    t = torch.linspace(0, 4*torch.pi, num_points, device=device).view(-1, 1)  # 시간 데이터
    t.requires_grad = True  # 자동 미분을 위해 requires_grad 설정
    y_true_1 = 1.0 * torch.sin(2.0 * torch.pi* t)
    y_true_2 = 3.0 * torch.sin(4.0 * torch.pi* t)
    y_true = y_true_1 + y_true_2
    return t, y_true

if Drawing:
    t_data, y_true = generate_data_total()
    plt.plot(t_data.cpu().detach().numpy(), y_true.cpu().detach().numpy(), 'b-', label='True')  # detach() 사용
    plt.plot(t_data.cpu().detach().numpy(), y_true.cpu().detach().numpy(), 'b*', label='True')  
    plt.show()

class HybridPINN(nn.Module):
    def __init__(self):
        super(HybridPINN, self).__init__()
        # CNN 레이어
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        self.activation = nn.Tanh()
        # 선형 레이어
        self.fc1 = nn.Linear(128, 50)
        self.fc2 = nn.Linear(50, 1)
      
    def forward(self, t):
        # Conv1d에 맞게 차원 조정
        t = t.unsqueeze(1).permute(0, 2, 1)  # (batch_size, channels, length)
        # CNN 레이어 통과
        x = self.activation(self.conv1(t))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        # 마지막 차원을 평균내서 축소
        x = torch.mean(x, dim=2)  # (batch_size, channels)
        # 선형 레이어를 통한 출력 계산
        x = self.activation(self.fc1(x))
        output = self.fc2(x)
        return output
    
# 잔차 계산 (예시로 y_pred와 a*sin(b*t)의 차이를 사용)
def residual_loss(y_pred, t):
    # 모델의 예측 값 y_pred에 대해 t에 대한 미분 계산
    dy_dt = torch.autograd.grad(outputs=y_pred, inputs=t, 
                                grad_outputs=torch.ones_like(y_pred), 
                                create_graph=True)[0]
    # a, b에 대한 미분은 별도로 계산하지 않음, 잔차 계산 (미분 방정식 기반)
    residual = dy_dt - (a * b * torch.cos(b *torch.pi* t)) - (c * d * torch.cos(d * torch.pi* t)) 
    return torch.mean(residual ** 2)

# 전체 손실 함수 정의 (DNN 손실 + PINN 손실 + 독립적인 규제 항)
def total_loss_function(y_pred, y_true, t, lambda_data=10.0, lambda_physics=0.1):
    # 데이터 손실 (DNN 손실)
    loss_data = torch.mean((y_pred - y_true) ** 2)
    # 물리적 손실 (PINN 손실)
    loss_physics = residual_loss(y_pred, t)
    # 전체 손실
    total_loss = lambda_data * loss_data + lambda_physics * loss_physics
    return total_loss
    

# 학습 준비
model = HybridPINN().to(device)  # 네트워크 깊이를 더 늘림
optimizer = optim.Adam([
    {'params': model.parameters()},
    {'params': [a], 'lr': 0.0005},
    {'params': [b], 'lr': 0.001},
    {'params': [c], 'lr': 0.0005},
    {'params': [d], 'lr': 0.001}
], lr=0.001)

# 데이터 준비
t_data, y_true = generate_data_total()

if Training: 
    # 학습 루프
    num_epochs = 100000
    for epoch in range(num_epochs):
    
        y_pred = model(t_data)
        loss = total_loss_function(y_pred, y_true, t_data)
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_([a, b, c, d], clip_value=1.0)
        optimizer.step()
   
        # 학습 과정 출력
        if epoch % 2000 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item()}, a = {a.item()}, b = {b.item()}, c = {c.item()}, d = {d.item()}')

    # 결과 출력
    print(f'최종 추정 값: a = {a.item()} (정답: 1), b = {b.item()} (정답: 2), c = {c.item()} (정답: 3), d = {d.item()} (정답: 4)')

    # 결과 시각화
    plt.plot(t_data.cpu().detach().numpy(), y_true.cpu().detach().numpy(), 'b-', label='True')  # detach() 사용
    plt.plot(t_data.cpu().detach().numpy(), model(t_data).cpu().detach().numpy(), 'r--', label='Predicted')
    plt.plot(t_data.cpu().detach().numpy(), y_true.cpu().detach().numpy(), 'b*', label='True')  
    plt.legend()
    plt.show()
