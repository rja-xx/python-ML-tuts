import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("resources/car.data")
print(data.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
cls = le.fit_transform(list(data['class']))
safety = le.fit_transform(list(data['safety']))
print(buying)

predict = 'class'


x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors = 8)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)
predicted = model.predict(x_test)

names = ["unacc", "acc", "good", "vgood"]
for x in range(len(predicted)):
    print("pred: ", names[predicted[x]], " Data: ", x_test[x], "Actual: ", names[y_test[x]])
    '''
    n = model.kneighbors([x_test[x]], 5, True)
    print("N: ", n)
    '''
