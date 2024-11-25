import csv

# Mở file để đọc dữ liệu
with open('dataset.csv', mode='r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    
    # Bỏ qua dòng tiêu đề (dòng đầu tiên)
    next(reader)
    
    # Đọc và in từng dòng dữ liệu từ dòng thứ 2 trở đi, bỏ qua cột đầu tiên
    for row in reader:
        print(row[1:])
