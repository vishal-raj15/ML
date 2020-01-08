
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import random

digits = load_digits()

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target, test_size=0.2)

model.fit(X_train, y_train)

a = model.score(X_test, y_test)

rno = random.randint(1,1797)

y_predicted = model.predict([digits.data[rno]])


print("predicted ",y_predicted)
print('\n')
print(" real data ",digits.target[rno])
plt.gray()
plt.matshow(digits.images[rno])
plt.show()

#  @ https://github.com/codebasics