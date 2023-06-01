from sklearn.cluster import KMeans
import numpy as np
import pickle


def correct_ratio(lst):
    lst = list(lst)
    setOfLst = set(lst)
    Demo_dict = {}
    for item in setOfLst:
        Demo_dict.update({item: lst.count(item)})

    return max([j for i, j in Demo_dict.items()])/len(lst)


with open('../dataOfNumber.pickle', 'rb') as f:
    lst = pickle.load(f)
train_x, train_y, test_x, test_y = lst

train_y = np.array(train_y)
kmeans = KMeans(n_clusters=10).fit(train_x)
result = []
for i in range(10):
    s = correct_ratio(kmeans.predict(train_x[train_y == i]))
    result.append(s)

print(np.mean(result)*100)

#print(train_x[train_y == 7].shape[0])
#print(kmeans.predict(train_x[train_y == 7]))
