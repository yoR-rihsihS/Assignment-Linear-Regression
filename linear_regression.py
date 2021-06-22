import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('dataset.csv')
print(dataset.shape)
dataset.head()

X = dataset['Head Size(cm^3)'].values
Y = dataset['Brain Weight(grams)'].values

x_mean = np.mean(X)
y_mean = np.mean(Y)

n = len(X)

numerator = 0
denominator = 0
for i in range(n):
    numerator += (X[i] - x_mean) * (Y[i] - y_mean)
    denominator += (X[i] - x_mean) ** 2
    
b1 = numerator / denominator
b0 = y_mean - (b1 * x_mean)

#printing the coefficient b0 and b1
print(b1, b0)

x_max = np.max(X) + 100
x_min = np.min(X) - 100

x = np.linspace(x_min, x_max, 1000)
y = b0 + b1 * x

plt.plot(x, y, color='#00ff00', label='Linear Regression')

plt.scatter(X, Y, color='#ff0000', label='Data Point')

plt.xlabel('Head Size (cm^3)')

plt.ylabel('Brain Weight (grams)')

plt.legend()
plt.show()

rmse = 0
for i in range(n):
    y_pred=  b0 + b1* X[i]
    rmse += (Y[i] - y_pred) ** 2
    
rmse = np.sqrt(rmse/n)
print(rmse)

sumofsquares = 0
sumofresiduals = 0

for i in range(n) :
    y_pred = b0 + b1 * X[i]
    sumofsquares += (Y[i] - y_mean) ** 2
    sumofresiduals += (Y[i] - y_pred) **2
    
score  = 1 - (sumofresiduals/sumofsquares)

print("Accuracy : ", score)