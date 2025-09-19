# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize weights w and bias b, set learning rate α and number of iterations.
2. Iterate:
   Compute predictions using sigmoid
   Calculate cost and gradients
   Update w and b using gradient descent
3.Predict class labels (ŷ ≥ 0.5 → 1 else 0).
4.Evaluate accuracy and plot decision boundary.


## Program:
```

Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: K DHANUSRI POOJA 
RegisterNumber:  212224040068


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=200, n_features=2, n_classes=2,
                           n_clusters_per_class=1, n_redundant=0, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

m, n = X_train.shape
w = np.zeros((n, 1))
b = 0
alpha = 0.1
iterations = 1000
y_train = y_train.reshape(-1, 1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

costs = []
for i in range(iterations):
    z = np.dot(X_train, w) + b
    y_hat = sigmoid(z)
    cost = -(1/m) * np.sum(y_train * np.log(y_hat) + (1 - y_train) * np.log(1 - y_hat))
    costs.append(cost)
    dw = (1/m) * np.dot(X_train.T, (y_hat - y_train))
    db = (1/m) * np.sum(y_hat - y_train)
    w -= alpha * dw
    b -= alpha * db

def predict(X, w, b):
    z = np.dot(X, w) + b
    return np.where(sigmoid(z) >= 0.5, 1, 0)

y_pred = predict(X_test, w, b)
accuracy = accuracy_score(y_test, y_pred)

print("Final Weights:", w)
print("Final Bias:", b)
print("Final Cost:", costs[-1])
print("Accuracy:", accuracy)

plt.figure(figsize=(8,6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', edgecolors='k')
x_boundary = np.linspace(min(X_test[:, 0]), max(X_test[:, 0]), 100)
y_boundary = -(w[0] * x_boundary + b) / w[1]
plt.plot(x_boundary, y_boundary, 'k', label="Decision Boundary")
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title("Logistic Regression (Gradient Descent)")
plt.show()

```

## Output:
<img width="1754" height="1037" alt="image" src="https://github.com/user-attachments/assets/d5688028-ff05-4b19-8b2c-f0d9b5dc91af" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

