import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pickle
import pydotplus


with open('../dataOfNumber.pickle', 'rb') as f:
    lst = pickle.load(f)
train_x, train_y, test_x, test_y = lst

clf = DecisionTreeClassifier(random_state=0)
clf.fit(train_x, train_y)
result = clf.predict(test_x)
test_y = np.array(test_y)
print(np.sum(result == test_y)/len(result)*100)
print(clf.get_depth())
dot_data = tree.export_graphviz(clf, filled=True, class_names=['0', '1', '2', '3', '4', '5', '6', '7','8', '9'])
# graph = pydotplus.graph_from_dot_data(dot_data)
# 保存图像到pdf文件
# graph.write_pdf('wande.pdf')