import numpy as np
import csv

# Softmax function
def softmax_function(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

# Read Weight and Bias from csv 
def load_weights_and_bias(filename='weight.csv'):
    W = []
    B = []

    # Read file CSV
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        
        header = next(reader)
        for row in reader:
            if row[0] == 'Bias':
                break
            W.append([float(value) for value in row])

        for row in reader:
            if row:  
                B.append(float(row[0]))

    W = np.array(W)
    B = np.array(B).reshape(-1, 1)

    return W, B

# Predict function
def predict_quality(X, W, B):
    Z = np.dot(W, X.T) + B 
    return softmax_function(Z)

W_loaded, B_loaded = load_weights_and_bias('weight.csv')

print("W và B đã được đọc từ 'weight.csv'.")

# Standardiza input function
def standardize_input(input_values):
    mean = [0, 0, 0]  
    std = [1, 1, 1]   

    # Chuẩn hóa dữ liệu
    return (input_values - mean) / std

# Hàm lấy input từ người dùng và thực hiện dự đoán
def get_input_and_predict():
    N = float(input("Nhập giá trị Nitrogen (N) (mg/kg): "))
    P = float(input("Nhập giá trị Phosphorus (P) (mg/kg): "))
    K = float(input("Nhập giá trị Potassium (K) (mg/kg): "))
    
    input_values = np.array([N, P, K])
    input_values = standardize_input(input_values)

    Y_pred = predict_quality(input_values.reshape(1, -1), W_loaded, B_loaded)
    
    # In kết quả dự đoán
    print("\nXác suất dự đoán cho từng lớp chất lượng:")
    print(Y_pred)
    
    # Lớp có xác suất cao nhất
    predicted_class = np.argmax(Y_pred)
    print(f"Lớp dự đoán: {predicted_class}")

# Gọi hàm để người dùng nhập dữ liệu và dự đoán
get_input_and_predict()
