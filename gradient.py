from sklearn.linear_model import LinearRegression
import numpy as np
import math

import matplotlib.pyplot as plt
x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
x=np.array(x)
y = [1,3,2,4,8,4,6,6,4,7,14,13,15,19,13,17,12,10,19,14]
x_ = np.array(x).reshape(-1,1)
y_ = np.array(y)

eta = 0.1
n=1000
m=100

theta = np.random.randn(2,1)

for i in range(n):
    gradient = 2/m * (x.T)*(x*theta - y)
    theta = theta - eta*gradient

for i in range(len(x)):
    x1 = [0,20]
    y1 = [x1[0]*theta[0][i] + theta[1][i] , x1[1]*theta[0][i] + theta[1][i]]
    plt.scatter(x,y)
    plt.plot(x1,y1)
    plt.show()


# error = 1/n((sum)(y-(mx + b))**2)

# de/dm = (2/n)*x(y-y')
# 
# eg .training over 10000 iteration
# m = m - learning_rate *de/dm
# b = b- learning_rate * de/db
# 
#  After each iteration we get minimum cost with keepin glow learning rate
# otherwise it may deflect from global minima 
# it is backpropogation where each weight and bias is getting updated every time
