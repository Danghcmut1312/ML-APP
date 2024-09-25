import numpy as np
import math

# Data
x = np.array([155, 180, 164, 162, 181, 182, 173, 190, 171, 170, 181, 182, 184, 209, 210])
y = np.array([51, 52, 54, 53, 55, 59, 61, 59, 63, 76, 64, 66, 69, 70, 80])

# Hypothesis function
def predict(x, Theta0, Theta1):
    return Theta0 + Theta1 * x

# Cost function
def cost(x, y, Theta0, Theta1):
    m = len(x)
    return (1 / (2 * m)) * np.sum((predict(x, Theta0, Theta1) - y) ** 2)

# Gradient descent với cập nhật giá trị theta
def gradient_descent(x, y, Theta0, Theta1, learning_rate, tolerance):
    m = len(x)
    cost_history = []
    iterations = 0
    previous_cost = float('inf')

    while True:
        h = predict(x, Theta0, Theta1)
        Gradient_Descent0 = Theta0 - learning_rate * (1 / m) * np.sum(h - y)
        Gradient_Descent1 = Theta1 - learning_rate * (1 / m) * np.sum((h - y) * x)
        Theta0, Theta1 = Gradient_Descent0, Gradient_Descent1
        
        current_cost = cost(x, y, Theta0, Theta1)
        cost_history.append(current_cost)
        
        if abs(previous_cost - current_cost) < tolerance:
            break
        
        previous_cost = current_cost
        iterations += 1

    return Theta0, Theta1, cost_history, iterations

# Khởi tạo giá trị ban đầu
Theta0 = 0
Theta1 = 0
learning_rate = 0.0001
tolerance = 1e-6

Theta0, Theta1, cost_history, iterations = gradient_descent(x, y, Theta0, Theta1, learning_rate, tolerance)

print(f"Theta0: {Theta0:.6f}, Theta1: {Theta1:.6f}")
print(f"Number of iterations: {iterations}")