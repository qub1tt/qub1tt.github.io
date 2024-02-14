---
read_time: true # calculate and show read time based on number of words
show_date: true # show the date of the post
title: Linear Regression
date: 2024-02-14
tags: [machine learning, math]
author: qub1tt
math: true
---

## Giới thiệu

Nếu đã từng học qua môn Xác suất thống kê (Probability and Statistics) chắc hẳn ai cũng biết về phương trình hồi quy tuyến tính và trong bài viết này sẽ đề cập đến ứng dụng của nó trong lĩnh vực Machine Learning.

Linear Regression là một phương pháp thống kê được sử dụng trong machine learning và thống kê để mô hình hóa mối quan hệ tuyến tính giữa một biến phụ thuộc (dependent variable) và một hoặc nhiều biến độc lập (independent variables). Mục tiêu của Linear Regression là tìm ra một đường thẳng (hay mặt phẳng trong không gian nhiều chiều) sao cho tổng bình phương sai số giữa giá trị dự đoán và giá trị thực tế là nhỏ nhất.

Công thức cơ bản của Linear Regression được biểu diễn như sau:
$\hat{y} = wx + b$

Trong đó:

- $\hat{y}$ là giá trị mô hình dự đoán
- $w$ và $b$ là hệ số góc (hằng số)
- $x$ là giá trị input

## Loss function

$L(\hat{y}) = (\hat{y} - y)^{2}$

Ý tưởng:

Hàm này sẽ so sánh giá trị $\hat{y}$ tức giá trị dự đoán và giá trị thực tế $y$ bằng cách tính bình phương khoảng cách giữa chúng. Nếu như giá trị L quá lớn tức là mô hình dự đoán chưa chính xác nó sẽ tự động cập nhật lại hai giá trị $w$ và $b$ để đưa ra một kết quả chính xác hơn.

Đây chính là phương pháp Gradient Descent:

![image](/assets/img/gradient_descent.png)

Đầu tiên ta sẽ lần lượt tính đạo hàm của $L$ theo $w$ và $b$ sau đó lấy giá trị ban đầu trừ đi giá trị đạo hàm vừa tính được nhân với $\eta$ (ở đây là learning rate - một tham số cần có của mô hình huấn luyện, tùy thuộc vào hàm loss ta sẽ cài đặt một giá trị phù hợp).

## Thí nghiệm bài toán

![image](/assets/img/ex1.png)

Giả sử ta cần huấn luyện mô hình dự đoán giá nhà dựa trên diện tích đất. Dataset trên gồm 4 mẫu, bắt đầu với mẫu thứ nhất ta sẽ tính được giá trị hàm loss cho ra 128.5 - một con số khá lớn, tiếp tục thực hiện phương pháp gradient descent đã nêu trên thì ta được kết quả:

![image](/assets/img/ex2.png)

Lúc này giá trị loss chỉ có 0.868 tức là mô hình đã dự đoán gần đúng

![image](/assets/img/ex3.png)

Ở đây cho thấy nếu như ta không cập nhật giá trị mới cho 2 biến $w$ và $b$ thì sẽ khiến cho mô hình dự đoán sai hoàn toàn và nếu làm như thông thường thì ta sẽ thấy các điểm đã gần như hội tụ trên một đường thẳng tức là mô hình của chúng ta có khả năng dự đoán ổn.

## Minh họa trên Python

Trước tiên, cần hai thư viện numpy và matplotlib

```python
import numpy as np
import matplotlib.pyplot as pl
```

Tiếp theo, khai báo và biểu diễn dữ liệu trên một đồ thị.

```python
areas  = [6.7, 4.6, 3.5, 5.5]
prices = [9.1, 5.9, 4.6, 6.7]
N = len(areas)

plt.scatter(areas, prices)
plt.xlabel('Area (x 100$m^2$)')
plt.ylabel('Price (Tael)')
plt.xlim(3,7)
plt.ylim(4,10)
plt.show()
```

![image](/assets/img/plot.png)

Cài đặt các hàm tính toán:

```python
# forward
def predict(x, w, b):
    return x*w + b

# compute gradient
def gradient(y_hat, y, x):
    dw = 2*x*(y_hat-y)
    db = 2*(y_hat-y)

    return (dw, db)

# update weights
def update_weight(w, b, lr, dw, db):
    w_new = w - lr*dw
    b_new = b - lr*db

    return (w_new, b_new)
```

Tiếp theo sẽ tiến hành việc huấn luyện, ở đây đặt giá trị b là 0.04, w là -0.34 và learning rate sẽ là 0.005. Ta huấn luyện trên 30 epochs:

```python
# data preparation
import numpy as np
import matplotlib.pyplot as plt

x_data = [6.7, 4.6, 3.5, 5.5]
y_data = [9.1, 5.9, 4.6, 6.7]

print(f'areas: {x_data}')
print(f'prices: {y_data}')
print(f'data_size: {N}')

# init weights
b = 0.04
w = -0.34
lr = 0.005

# parameter
epoch_max = 30
losses = [] # monitoring

for epoch in range(epoch_max):
    # shuffle data


    # for an epoch
    for i in range(N):

        # get a sample
        x = x_data[i]
        y = y_data[i]

        # predict y_hat
        y_hat = predict(x, w, b)

        # compute loss
        loss = (y_hat-y)*(y_hat-y)

        # for debug
        losses.append(loss)
        print(loss)

        # compute gradient
        (dw, db) = gradient(y_hat, y, x)

        # update weights
        (w, b) = update_weight(w, b, lr, dw, db)

```

Kết quả sau khi train:

![image](/assets/img/res.png)

Cuối cùng vẽ lại đồ thị sau khi train mô hình:

```python
y_data = [x*w + b for x in x_data]
plt.plot(x_data, y_data, 'r')
plt.scatter(areas, prices)
```

![image](/assets/img/matplot.png)

Ở trên chỉ đề cập cách cài đặt thuật toán linear regression cơ bản, ta có thể sử dụng thư viện có sẵn của python để sử dụng các thuật toán ML.
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

## Tổng kết

Linear Regression là một mô hình đơn giản và nó được dùng để giải quyết mỗi bài toán tuyến tính, không thể biễu diễn được các mô hình phức tạp như các bài toán phi tuyến tính. Mặc dù vậy đây là một trong những thuật toán cơ bản của ML nhằm tạo nền tảng để các nhà nghiên cứu phát triển cho Deep Learning và AI sau này.

![image](/assets/img/summary.png)
