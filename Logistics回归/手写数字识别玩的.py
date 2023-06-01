# 用logistics试试数字图片分类，
# 用saga有96.93，lbfgs有97.15，因为误差函数是个凸函数，用牛顿类的肯定效果好，速度快几个量级
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle


with open('../dataOfNumber.pickle', 'rb') as f:
    lst = pickle.load(f)
train_x, train_y, test_x, test_y = lst

clf = LogisticRegression(solver='lbfgs', max_iter=100, multi_class='multinomial')
clf.fit(train_x, train_y)
right_vector = clf.predict(test_x) == test_y
print(np.sum(right_vector)/len(right_vector)*100)
