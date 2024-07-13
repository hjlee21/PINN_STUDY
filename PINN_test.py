import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt

# 1. 데이터 생성
def true_solution(c, c_pi, t):
    return c*np.sin(c_pi * np.pi * t)

# 2. PINN 모델 정의
class PINN(Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.dense1 = layers.Dense(32, activation='tanh')
        self.dense2 = layers.Dense(32, activation='tanh')
        self.dense3 = layers.Dense(32, activation='tanh')
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
