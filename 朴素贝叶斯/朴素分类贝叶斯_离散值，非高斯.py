import numpy as np
from sklearn.naive_bayes import CategoricalNB
import pickle


with open('../dataOfNumber.pickle', 'rb') as f:
    lst = pickle.load(f)
train_x, train_y, test_x, test_y = lst

train_x[0] = np.ones((1, 1024))  # x1特征有3个类，x2有5个，根据原理，最多只需要添加五条数据即可
clf = CategoricalNB()
clf.fit(train_x, train_y)
# print(clf.predict(test_x))
#print(np.sum(clf.predict(test_x) == test_y) / len(test_y)*100)
print(test_x[[2]])
print(test_x[2])
# print(clf.predict(test_x[[600]]))
# rint(clf.predict_log_proba(test_x[[600]]))

