import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn
from sklearn.linear_model import LinearRegression

list_data = []
for i in range(1000):
    x = np.random.normal(0.0, 0.55)
    y = x * 0.5 + 0.6 + np.random.normal(0.0,0.02)
    list_data.append([x,y])
# print(list_data)

X_data = [s[0] for s in list_data]
Y_data = [s[1] for s in list_data]

LR = LinearRegression()
X = np.array(X_data).reshape(-1,1)
Y = np.array(Y_data).reshape(-1,1)
LR.fit(X,Y)
score = LR.score(X,Y)
# print(LR.coef_)
# print(LR.intercept_)
X_predict = np.random.normal(0.0,0.1,size=(5,1))
print(X_predict)
print(LR.predict(X_predict))
# print(score)

# from sklearn.datasets import load_iris
# from sklearn.linear_model import LogisticRegression
# X, y = load_iris(return_X_y=True)
# clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X, y)
# print(clf.predict(X[:2, :]))
# print(clf.predict_proba(X[:2, :]))
# print(clf.score(X, y))
