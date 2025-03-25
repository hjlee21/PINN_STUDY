import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

# 학습 가능한 변수 a와 b를 정의
a = torch.nn.Parameter(torch.tensor(10.0)) # 초기값 설정
b = torch.nn.Parameter(torch.tensor(10.0)) # 초기값 설정

# 데이터 생성 (예시 데이터)
def generate_data(num_points=100):
    t = torch.linspace(0, 8, num_points).view(-1, 1)  # 시간 데이터
    t.requires_grad = True  # 자동 미분을 위해 requires_grad 설정
    # 실제 값 생성: a = 2.0, b = 3.0으로 가정
    y_true = 2.0 * torch.sin(3.0 * t)
    return t, y_true

class DeepPINN(nn.Module):
    def __init__(self):
        super(DeepPINN, self).__init__()
        # 입력층
        self.input_layer = nn.Linear(1, 50)
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(50, 50),
            nn.Tanh()
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(50, 50),
            nn.Tanh()
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(50, 50),
            nn.Tanh()
        )
        # 출력층
        self.output_layer = nn.Linear(50, 1)
        # 활성화 함수
        self.activation = nn.Tanh()
      
        
    def forward(self, t):
        # 입력층
        x = self.activation(self.input_layer(t))
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        # 출력층
        output = self.output_layer(x)
        return output
    
# 잔차 계산 (예시로 y_pred와 a*sin(b*t)의 차이를 사용)
def residual_loss(y_pred, t):
    # 모델의 예측 값 y_pred에 대해 t에 대한 미분 계산
    dy_dt = torch.autograd.grad(outputs=y_pred, inputs=t, 
                                grad_outputs=torch.ones_like(y_pred), 
                                create_graph=True)[0]
    # a, b에 대한 미분은 별도로 계산하지 않음, 잔차 계산 (미분 방정식 기반)
    residual = dy_dt - (a * b * torch.cos(b * t))
    return torch.mean(residual ** 2)

# 전체 손실 함수 정의 (DNN 손실 + PINN 손실)
def total_loss_function(y_pred, y_true, t, lambda_data=1.0, lambda_physics=0.1):
    # 데이터 손실 (DNN 손실)
    loss_data = torch.mean((y_pred - y_true) ** 2)
    # 물리적 손실 (PINN 손실)
    loss_physics = residual_loss(y_pred, t)
    # 전체 손실
    total_loss = lambda_data * loss_data + lambda_physics * loss_physics
    return total_loss
    

# 학습 준비
model = DeepPINN()  # 네트워크 깊이를 더 늘림
optimizer = optim.Adam([{'params': model.parameters()}, {'params': [a, b]}], lr=0.0005)

# 데이터 준비
t_data, y_true = generate_data()

# 학습 루프
num_epochs = 100000
for epoch in range(num_epochs):
    # 순전파 계산
    y_pred = model(t_data)

    # 전체 손실 계산
    loss = total_loss_function(y_pred, y_true, t_data)

    # 역전파 및 최적화
    optimizer.zero_grad()
    loss.backward(retain_graph=True)  # retain_graph=True 설정
    optimizer.step()

    # 학습 과정 출력
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}: Loss = {loss.item()}, a = {a.item()}, b = {b.item()}')

# 결과 출력
print(f'최종 추정 값: a = {a.item()} (정답: 2), b = {b.item()} (정답: 3)')

# 결과 시각화
plt.plot(t_data.detach().numpy(), y_true.detach().numpy(), 'bo', label='True')  # detach() 사용
plt.plot(t_data.detach().numpy(), model(t_data).detach().numpy(), 'r--', label='Predicted')
plt.legend()
plt.show()
