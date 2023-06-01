import numpy as np
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt


X = np.array([[1, 1],
              [2, 2.2],
              [3, 2.7],
              [12, 11.8],
              [4, 5],
              [7, 7.1]])
#with open('data.pickle', 'rb') as f:
#    X = pickle.load(f)

plt.scatter(X[:, 0], X[:, 1])
plt.show()
pca = PCA(n_components=2)
pca.fit(X)

print(pca.explained_variance_ratio_)  # 每个主成分解释的比例，即数据协方差矩阵特征所占总的大小
print(pca.explained_variance_)  # 每个主成分的方差，即对应的特征值
print(pca.components_)  # 每一行hang是对应的主成分向量，即特征向量

print(pca.fit_transform(X))
#Y = np.array([[100, 100, 1],
#              [50, 1, 2]])
#print(pca.transform(Y))

