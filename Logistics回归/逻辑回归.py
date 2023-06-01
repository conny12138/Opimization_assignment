import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

df = pd.read_table('horseColicTraining.txt', header=None)
clf = LogisticRegression(solver='saga', max_iter=500, multi_class='multinomial')
clf.fit(df.iloc[:, :-1], df.iloc[:, -1])

test_df = pd.read_table('horseColicTest.txt', header=None)
right_vector = clf.predict(test_df.iloc[:, :-1]) == test_df.iloc[:, -1]
print(np.sum(right_vector)/len(right_vector)*100)
