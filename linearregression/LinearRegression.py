import pandas as pd
import numpy as np
import sklearn

from sklearn import linear_model
import pickle
from matplotlib import style
import matplotlib.pyplot as pyplot


data = pd.read_csv('resources/student-mat.csv', sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"
data.drop([predict],1)
X = np.array(data.drop(['G3'], 1))
Y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print('coeff: ', linear.coef_)
print('intercept: ', linear.intercept_)
with open('resources/linear_model.pickle', 'wb') as f:
  pickle.dump( linear, f)
with open('resources/linear_model.pickle', 'wb') as f:
  pickle.dump( linear, f)
best = 0
'''for _ in range(30):
  x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)
  linear.fit(x_train, y_train)
  linear = linear_model.LinearRegression()
  linear.fit(x_train, y_train)
  acc =  linear.score(x_test, y_test)
  print(best)
  if(acc > best):
    best = acc
    with open('linear_model.pickle', 'wb') as f:
      pickle.dump( linear, f)
'''

pickle_in = open('resources/linear_model.pickle', 'rb')
linear = pickle.load(pickle_in)
predictions = linear.predict(x_test)
for x in range(len(predictions)):
  print(predictions[x], x_test[x], y_test[x])

p='G2'
style.use('ggplot')
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel('Final grade')
pyplot.show()