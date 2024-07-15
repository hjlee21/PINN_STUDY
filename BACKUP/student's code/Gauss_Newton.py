import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.fft import fft, fftfreq

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

# Calculate sampling interval
T = np.mean(np.diff(x_data))  # Average sampling interval
N = len(x_data)  # Number of data points

print(f"Sampling interval T: {T}")
print(f"Number of data points N: {N}")

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
a = y_data.max() - y_data.mean()
b = initial_b
c = np.mean(y_data)

# Gauss-Newton algorithm
max_iter = 1000
tol = 1e-6
lambda_reg = 1e-5  # Regularization parameter

for _ in range(max_iter):
    # Calculate residuals
    r = y_data - (a * np.sin(b * x_data) + c)

    # Calculate Jacobian
    J = np.zeros((len(x_data), 3))
    J[:, 0] = np.sin(b * x_data)
    J[:, 1] = a * x_data * np.cos(b * x_data)
    J[:, 2] = 1  # Derivative with respect to c

    # Gauss-Newton step with regularization
    JTJ = J.T @ J
    JTJ_reg = JTJ + lambda_reg * np.eye(JTJ.shape[0])  # Add small value to diagonal
    delta = np.linalg.inv(JTJ_reg) @ J.T @ r
    a += delta[0]
    b += delta[1]
    c += delta[2]

    # Check for convergence
    if np.linalg.norm(delta) < tol:
        break

# Output results
print(f"Estimated parameters: a = {a}, b = {b}, c = {c}")

# Visualize results
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_data, a * np.sin(b * x_data) + c, label='Fitted', color='red')
plt.legend()
plt.show()
