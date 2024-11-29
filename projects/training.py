import numpy as np
import csv
from sklearn.preprocessing import StandardScaler

# Read data from CSV file
with open('dataset.csv', mode='r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    
    # Skip the first row (header)
    next(reader)
    
    # Create lists to hold values from the columns
    column_2 = []  # Nitrogen (N) mg/kg
    column_3 = []  # Phosphorus (P) mg/kg
    column_4 = []  # Potassium (K) mg/kg
    column_5 = []  # Rating scale

    # Read and add data to the lists
    for row in reader:
        column_2.append(int(row[1]))  # Column 2: Nitrogen (N) mg/kg
        column_3.append(int(row[2]))  # Column 3: Phosphorus (P) mg/kg
        column_4.append(int(row[3]))  # Column 4: Potassium (K) mg/kg
        column_5.append(row[4])       # Column 5: Rating scale

# Convert data into numpy arrays
N = np.array(column_2)
P = np.array(column_3)
K = np.array(column_4)
QC = np.array(column_5)

# Encode the rating scale into numerical values
labels = {'Thấp, nghèo': 0, 'Trung bình': 1, 'Trung bình đến giàu': 2, 'Rất giàu': 3}
y = np.array([labels[qc] for qc in QC])

# Combine features into the matrix X
X = np.column_stack((N, P, K))  # Feature matrix X (with 3 features)

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Tạo ma trận trọng số W
W = np.random.randn(4, 3)

# Tạo vector bias B
B = np.random.randn(4, 1)

# Softmax function
def softmax_function(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=1)

# Predict function
def predict(X, B, W):
    Z = np.dot(W.T, X) + B
    return softmax_function(Z)

# Cost function
def cross_entropy_loss(X, Y, B, W):
    m = len(Y)
    epsilon = 1e-15
    y_pred = predict(X, B, W)
    cost = - np.sum(Y * np.log(y_pred + epsilon)) / m
    return cost

# Hàm tính gradient của W và B
def compute_gradients(X, Y, Y_pred):
    m = X.shape[1]
    dZ = Y_pred - Y  # Sai số giữa dự đoán và nhãn thực tế
    dW = np.dot(dZ, X.T) / m  # Gradient của W
    dB = np.sum(dZ, axis=1, keepdims=True) / m  # Gradient của B
    return dW, dB

# Hàm cập nhật trọng số và bias bằng Gradient Descent
def gradient_descent(X, Y, W, B, learning_rate, num_iterations):
    for i in range(num_iterations):
        # Tính giá trị z và xác suất dự đoán
        z = compute_z(X, W, B)
        Y_pred = softmax(z)

        # Tính hàm chi phí
        loss = cross_entropy_loss(Y, Y_pred)

        # Tính gradient
        dW, dB = compute_gradients(X, Y, Y_pred)

        # Cập nhật tham số W và B
        W -= learning_rate * dW
        B -= learning_rate * dB

        # In loss mỗi 100 lần lặp (có thể thay đổi theo nhu cầu)
        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss}")

    return W, B