from sklearn.linear_model import LinearRegression
import numpy as np

import matplotlib.pyplot as plt
x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
y = [1,3,2,4,8,4,6,6,4,7,14,13,15,19,13,17,12,10,19,14]
x_ = np.array(x).reshape(-1,1)
y_ = np.array(y)

lin_reg = LinearRegression()

lin_reg.fit(x_,y_)
a = lin_reg.intercept_
b = lin_reg.coef_

print( a, b)
x1=[0,20]

y1 = [b*x1[0] + a ,b*x1[1] + a]

plt.scatter(x,y)
plt.plot(x1,y1 , color='r')
plt.show()




