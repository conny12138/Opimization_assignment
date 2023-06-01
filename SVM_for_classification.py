from sklearn import svm
import pickle
import numpy as np
# 正确率98.52


with open('../dataOfNumber.pickle', 'rb') as f:
    lst = pickle.load(f)
train_x, train_y, test_x, test_y = lst

clf = svm.SVC()
clf.fit(train_x, train_y)

right_vector = clf.predict(test_x) == test_y
print(np.sum(right_vector)/len(right_vector)*100)
