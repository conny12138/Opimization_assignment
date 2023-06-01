# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pydotplus
from sklearn import tree


df = pd.read_table('lenses.txt', header=None)
order = {}
encoder = LabelEncoder()
for i in range(5):
    encoder.fit(df[i])
    order[i] = encoder.classes_
    df[i] = encoder.transform(df[i])
print(order)
#print(df)
# print(df.iloc[:, 0:4])

clf = DecisionTreeClassifier(random_state=0)
clf.fit(df.iloc[:, 0:4], df[4])
dot_data = tree.export_graphviz(clf, filled=True, class_names=order[4])
print(clf.predict([[2, 1, 1, 1], [0, 1, 0, 1]]))

#graph = pydotplus.graph_from_dot_data(dot_data)
# 保存图像到pdf文件
#graph.write_pdf('x.pdf')



