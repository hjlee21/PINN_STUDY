# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from scipy.fft import fft, fftfreq
#
# import pandas as pd
# import os
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
#
# # Function to load and concatenate all CSV files in a given directory and its subdirectories
# def load_all_csv_files(directory):
#     all_files = []
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.endswith('.csv'):
#                 all_files.append(os.path.join(root, file))
#             else:
#                 print(f"Ignored file: {os.path.join(root, file)}")
#
#     print(f"Found {len(all_files)} CSV files.")
#
#     if not all_files:
#         raise ValueError("No CSV files were found.")
#
#     df_list = []
#     for file in all_files:
#         try:
#             df = pd.read_csv(file)
#             df_list.append(df)
#             print(f"Loaded file: {file}, shape: {df.shape}")
#         except Exception as e:
#             print(f"Error loading {file}: {e}")
#
#     if not df_list:
#         raise ValueError("No CSV files were loaded.")
#
#     return pd.concat(df_list, ignore_index=True)
#
# # Load and preprocess data
# directory = r"D:\Work_KAIST\Signal_Data_Parser\Normal_Data\filtered_data"  # Replace with your directory path
# ###########################################################
# ## make sure to change the normalisation range of y data when changing the directory ##
# ###########################################################
# data = load_all_csv_files(directory)
#
# # Prepare input variables for training
# x_data = data['Time Elapsed'].values.astype(np.float32)
# y_data = data['Probe 10'].values.astype(np.float32)
#
# # # Normalize the data
# # scaler = MinMaxScaler()
# # x_data = scaler.fit_transform(x_data.reshape(-1, 1)).flatten()
# # # y_data = scaler.fit_transform(y_data.reshape(-1, 1)).flatten()
# #
# # # Normalize y_data to range [-1, 1]
# # y_scaler = MinMaxScaler(feature_range=(-1, 1))
# # y_data = y_scaler.fit_transform(y_data.reshape(-1, 1)).flatten()
#
# # Calculate sampling interval
# T = np.mean(np.diff(x_data))  # Average sampling interval
# N = len(x_data)  # Number of data points
#
# # Perform FFT
# yf = fft(y_data)
# xf = fftfreq(N, T)[:N//2]  # Only consider positive frequency components
#
# # Compute magnitude of FFT results
# fft_magnitude = np.abs(yf[:N//2])
#
# # Plot FFT result to understand the frequency components
# plt.figure(figsize=(10, 6))
# plt.plot(xf, fft_magnitude)
# plt.title('FFT of y_data')
# plt.xlabel('Frequency')
# plt.ylabel('Magnitude')
# plt.grid(True)
# plt.show()
#
# # Ignore the zero frequency component and find the dominant frequency
# fft_magnitude[0] = 0  # Ignore the DC component
# dominant_frequency = xf[np.argmax(fft_magnitude)]
# initial_b = 2 * np.pi * dominant_frequency  # Convert to angular frequency
#
# print(f"Dominant frequency: {dominant_frequency}")
# print(f"Initial guess for b (angular frequency): {initial_b}")
#
# # Initial parameters
# a_initial = y_data.max() - y_data.mean()
# b_initial = initial_b
# c_initial = np.mean(y_data)
#
# # 모델 정의
# class SinModel(Model):
#     def __init__(self, a_initial, b_initial, c_initial):
#         super(SinModel, self).__init__()
#         self.a = tf.Variable(initial_value=a_initial, trainable=True, dtype=tf.float32)
#         self.b = tf.Variable(initial_value=b_initial, trainable=True, dtype=tf.float32)
#         self.c = tf.Variable(initial_value=c_initial, trainable=True, dtype=tf.float32)
#
#     def call(self, x):
#         return self.a, self.b, self.c
#
# # SinModel 인스턴스 생성
# model = SinModel(a_initial, b_initial, c_initial)
#
# # 손실 함수 정의
# def custom_loss(y_true, y_pred, a, b, c, x):
#     # Predicted final value based on a * sin(b * x) + c
#     y_pred_final = a * tf.sin(b * x) + c
#
#     # Mean Squared Error (MSE) Loss
#     mse_loss = tf.reduce_mean(tf.square(y_true - y_pred_final))
#
#     # L2 Regularization to prevent 'a' from becoming too small
#     reg_loss = tf.abs((1 - tf.abs(a)))
#
#     # Return combined loss
#     return mse_loss + 0.001 * reg_loss
#
# # 옵티마이저 정의
# optimizer = Adam(learning_rate=0.001)
#
# # 훈련
# def train_step(x, y):
#     with tf.GradientTape() as tape:
#         a, b, c = model(x)
#         y_pred = a * tf.sin(b * x) + c
#         loss = custom_loss(y, y_pred, a, b, c, x)
#     trainable_vars = [model.a, model.b, model.c]
#     gradients = tape.gradient(loss, trainable_vars)
#     if None in gradients:
#         raise ValueError("None value found in gradients")
#     optimizer.apply_gradients(zip(gradients, trainable_vars))
#     return loss
#
# # 데이터 텐서로 변환
# x_tensor = tf.convert_to_tensor(x_data, dtype=tf.float32)
# y_tensor = tf.convert_to_tensor(y_data, dtype=tf.float32)
#
# # 모델 빌드 및 첫 호출
# model.build(input_shape=(None,))
# model(x_tensor[:1])  # 모델을 한번 호출하여 가중치를 초기화
#
# # 모델의 학습 가능한 변수 확인
# print("Trainable variables before training:", model.trainable_variables)
#
# # 모델 훈련
# a_values = []
# b_values = []
# c_values = []
# epochs = 1000
# for epoch in range(epochs):
#     loss = train_step(x_tensor, y_tensor)
#     if epoch % 100 == 0:
#         print(f"Epoch {epoch}, Loss: {loss.numpy()}")
#         a, b, c = model(x_tensor)
#         a_values.append(a.numpy())
#         b_values.append(b.numpy())
#         c_values.append(c.numpy())
#         print(f"Trained a: {a.numpy()}, Trained b: {b.numpy()}, Trained c: {c.numpy()}")
#
# # 결과 출력
# a, b, c = model(x_tensor)
#
# # 예측 값 계산
# y_pred = a * tf.sin(b * x_tensor) + c
#
# # a, b, c 값 플로팅
# plt.figure(figsize=(10, 6))
# plt.plot(range(0, epochs, 100), a_values, label='a values')
# plt.plot(range(0, epochs, 100), b_values, label='b values')
# plt.plot(range(0, epochs, 100), c_values, label='c values')
# plt.xlabel('Epoch')
# plt.ylabel('Value')
# plt.title('a, b, and c values over Epochs')
# plt.legend()
# plt.show()
#
# # 플로팅
# plt.figure(figsize=(10, 6))
# plt.plot(x_data, y_data, label='Original Data')
# plt.plot(x_data, y_pred, label='Predicted Data', linestyle='--')
# plt.xlabel('Test_Time')
# plt.ylabel('Test_Value')
# plt.title('Original vs Predicted Data')
# plt.legend()
# plt.show()

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from scipy.fft import fft, fftfreq

import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Function to load and concatenate all CSV files in a given directory and its subdirectories
def load_all_csv_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                all_files.append(os.path.join(root, file))
            else:
                print(f"Ignored file: {os.path.join(root, file)}")

    print(f"Found {len(all_files)} CSV files.")

    if not all_files:
        raise ValueError("No CSV files were found.")

    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
            print(f"Loaded file: {file}, shape: {df.shape}")
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if not df_list:
        raise ValueError("No CSV files were loaded.")

    return pd.concat(df_list, ignore_index=True)

# Load and preprocess data
directory = r"D:\Work_KAIST\Signal_Data_Parser\Normal_Data\filtered_data"  # Replace with your directory path
###########################################################
## make sure to change the normalisation range of y data when changing the directory ##
###########################################################
data = load_all_csv_files(directory)

# Prepare input variables for training
x_data = data['Time Elapsed'].values.astype(np.float32)
y_data = data['Probe 10'].values.astype(np.float32)

# # Normalize the data
# scaler = MinMaxScaler()
# x_data = scaler.fit_transform(x_data.reshape(-1, 1)).flatten()
#
# # Normalize y_data to range [-1, 1]
# y_scaler = MinMaxScaler(feature_range=(-1, 1))
# y_data = y_scaler.fit_transform(y_data.reshape(-1, 1)).flatten()

# Calculate sampling interval
T = np.mean(np.diff(x_data))  # Average sampling interval
N = len(x_data)  # Number of data points

# Perform FFT
yf = fft(y_data)
xf = fftfreq(N, T)[:N//2]  # Only consider positive frequency components

# Compute magnitude of FFT results
fft_magnitude = np.abs(yf[:N//2])

# Plot FFT result to understand the frequency components
plt.figure(figsize=(10, 6))
plt.plot(xf, fft_magnitude)
plt.title('FFT of y_data')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()

# Ignore the zero frequency component and find the dominant frequency
fft_magnitude[0] = 0  # Ignore the DC component
dominant_frequency = xf[np.argmax(fft_magnitude)]
initial_b = 2 * np.pi * dominant_frequency  # Convert to angular frequency

print(f"Dominant frequency: {dominant_frequency}")
print(f"Initial guess for b (angular frequency): {initial_b}")

# Initial parameters
a_initial = np.abs(y_data.max() - y_data.mean())
a_sign = np.sign(y_data.max() - y_data.mean())
b_initial = initial_b
c_initial = np.mean(y_data)

# 모델 정의
class SinModel(Model):
    def __init__(self, a_initial, b_initial, c_initial):
        super(SinModel, self).__init__()
        self.a = tf.Variable(initial_value=a_initial, trainable=True, dtype=tf.float32)
        self.b = tf.Variable(initial_value=b_initial, trainable=True, dtype=tf.float32)
        self.c = tf.Variable(initial_value=c_initial, trainable=True, dtype=tf.float32)

    def call(self, x):
        return self.a, self.b, self.c

# SinModel 인스턴스 생성
model = SinModel(a_initial, b_initial, c_initial)

# 손실 함수 정의
def custom_loss(y_true, y_pred, a, b, c, x, a_sign):
    # Predicted final value based on a * sin(b * x) + c
    y_pred_final = a_sign * a * tf.sin(b * x) + c

    # Mean Squared Error (MSE) Loss
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred_final))

    # L2 Regularization to prevent 'a' from becoming too small
    reg_loss = tf.abs((1 - tf.abs(a)))

    # Return combined loss
    return mse_loss + 0.001 * reg_loss

# 옵티마이저 정의
optimizer = Adam(learning_rate=0.001)

# 훈련
def train_step(x, y, a_sign):
    with tf.GradientTape() as tape:
        a, b, c = model(x)
        y_pred = a_sign * a * tf.sin(b * x) + c
        loss = custom_loss(y, y_pred, a, b, c, x, a_sign)
    trainable_vars = [model.a, model.b, model.c]
    gradients = tape.gradient(loss, trainable_vars)
    if None in gradients:
        raise ValueError("None value found in gradients")
    optimizer.apply_gradients(zip(gradients, trainable_vars))
    return loss

# 데이터 텐서로 변환
x_tensor = tf.convert_to_tensor(x_data, dtype=tf.float32)
y_tensor = tf.convert_to_tensor(y_data, dtype=tf.float32)

# 모델 빌드 및 첫 호출
model.build(input_shape=(None,))
model(x_tensor[:1])  # 모델을 한번 호출하여 가중치를 초기화

# 모델의 학습 가능한 변수 확인
print("Trainable variables before training:", model.trainable_variables)

# 모델 훈련
a_values = []
b_values = []
c_values = []
epochs = 10000
for epoch in range(epochs):
    loss = train_step(x_tensor, y_tensor, a_sign)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")
        a, b, c = model(x_tensor)
        a_values.append(a.numpy())
        b_values.append(b.numpy())
        c_values.append(c.numpy())
        print(f"Trained a: {a.numpy()}, Trained b: {b.numpy()}, Trained c: {c.numpy()}")

# 결과 출력
a, b, c = model(x_tensor)

# 예측 값 계산
y_pred = a_sign * a * tf.sin(b * x_tensor) + c

# a, b, c 값 플로팅
plt.figure(figsize=(10, 6))
plt.plot(range(0, epochs, 100), a_values, label='a values')
plt.plot(range(0, epochs, 100), b_values, label='b values')
plt.plot(range(0, epochs, 100), c_values, label='c values')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('a, b, and c values over Epochs')
plt.legend()
plt.show()

# 플로팅
plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, label='Original Data')
plt.plot(x_data, y_pred, label='Predicted Data', linestyle='--')
plt.xlabel('Test_Time')
plt.ylabel('Test_Value')
plt.title('Original vs Predicted Data')
plt.legend()
plt.show()

