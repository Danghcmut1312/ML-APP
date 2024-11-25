import csv
import numpy as np

# Open file dataset
with open('dataset.csv', mode='r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    
    # Skip first row in file
    next(reader)
    
    # Create arrays
    column_2 = []
    column_3 = []
    column_4 = []
    column_5 = []
    
    # Read and append data to array
    for row in reader:
        column_2.append(int(row[1]))  # Cột 2: Đạm (N) mg/kg
        column_3.append(int(row[2]))  # Cột 3: Lân (P) mg/kg
        column_4.append(int(row[3]))  # Cột 4: Kali (K) mg/kg
        column_5.append(row[4])       # Cột 5: Thang đánh giá
    
# Convert to numpy array
N = np.array(column_2)
P = np.array(column_3)
K = np.array(column_4)
QC = np.array(column_5)


# # Mở file để ghi dữ liệu
# with open('weight.csv', mode='a', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)

#     # Ghi dữ liệu từ các biến
#     writer.writerow([stt, so1, so2, tong])
