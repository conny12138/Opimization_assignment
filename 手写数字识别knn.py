# training188,197,194,198,185,186,194,200,179,203
# test86,96,91,84,113,107,86,95,90,88
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle


def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    with open(filename) as f:
        for i in range(32):
            lineStr = f.readline()
            for j in range(32):
                returnVect[0, 32*i+j] = int(lineStr[j])
        return returnVect


lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
lst1 = [188, 197, 194, 198, 185, 186, 194, 200, 179, 203]
# lst1 = [86, 96, 91, 84, 113, 107, 86, 95, 90, 88]
lst1 = [i+1 for i in lst1]

x = np.zeros((1, 1024))
y = []
for i in lst:
    for j in range(lst1[i]):
        # s = f'testDigits/{i}_{j}.txt'
        s = f'trainingDigits/{i}_{j}.txt'
        x = np.vstack((x, img2vector(s)))
        y.append(i)
x = x[1:, :]  # x矩阵数据集彻底处理好了这儿
# print(x)

classifier = KNeighborsClassifier(n_neighbors=7, algorithm='brute')
classifier.fit(x, y)  # 分类器在这儿训练完了

lst1 = [86, 96, 91, 84, 113, 107, 86, 95, 90, 88]
lst1 = [i+1 for i in lst1]
testing = np.zeros((1, 1024))
result = []
for i in lst:
    for j in range(lst1[i]):
        s = f'testDigits/{i}_{j}.txt'
        testing = np.vstack((testing, img2vector(s)))
        result.append(i)
result = np.array(result)
testing = testing[1:, :]

#with open('dataOfNumber.pickle', 'wb') as f:
#    pickle.dump([x, y, testing, result], f)

print(np.sum(result == classifier.predict(testing))/len(result)*100)
print(classifier.predict(testing).shape)
# 正确率97.57
