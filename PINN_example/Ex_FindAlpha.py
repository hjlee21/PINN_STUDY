import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt

# 1. 데이터 생성
def true_solution(x, t, alpha):
    return np.exp(-alpha * np.pi**2 * t) * np.sin(np.pi * x)

# 시간 및 공간 변수 설정
x = np.linspace(0, 1, 100)[:, None]
t = np.linspace(0, 1, 100)[:, None]
X, T = np.meshgrid(x, t)

# 실제 솔루션 (데이터 생성)
alpha_true = 0.01
u_true = true_solution(X, T, alpha_true)

# 2. PINN 모델 정의
class PINN(Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.dense1 = layers.Dense(20, activation='tanh')
        self.dense2 = layers.Dense(20, activation='tanh')
        self.dense3 = layers.Dense(20, activation='tanh')
        self.dense4 = layers.Dense(1, activation=None)
        self.alpha = tf.Variable(0.1, dtype=tf.float32)  # 초기 알파 값

    def call(self, inputs):
        x, t = inputs
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        t = tf.convert_to_tensor(t, dtype=tf.float32)
        u = self.dense1(tf.concat([x, t], axis=1))
        u = self.dense2(u)
        u = self.dense3(u)
        u = self.dense4(u)
        return u

# 3. 손실 함수 정의
def loss_function(model, x, t, u):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, t])
        u_pred = model([x, t])
        u_x = tape.gradient(u_pred, x)
        u_xx = tape.gradient(u_x, x)
        u_t = tape.gradient(u_pred, t)
    
    alpha = model.alpha
    f_pred = u_t - alpha * u_xx
    loss_data = tf.reduce_mean(tf.square(u - u_pred))
    loss_pde = tf.reduce_mean(tf.square(f_pred))
    loss = loss_data + loss_pde
    return loss

# 4. 모델 학습
model = PINN()
optimizer = tf.keras.optimizers.Adam()

x_tf = tf.convert_to_tensor(X.flatten()[:, None], dtype=tf.float32)
t_tf = tf.convert_to_tensor(T.flatten()[:, None], dtype=tf.float32)
u_tf = tf.convert_to_tensor(u_true.flatten()[:, None], dtype=tf.float32)

epochs = 10000
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss = loss_function(model, x_tf, t_tf, u_tf)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.numpy()}, Alpha: {model.alpha.numpy()}')

# 결과 출력
print(f'Estimated alpha: {model.alpha.numpy()}')