import numpy as np
import matplotlib.pyplot as plt


a = np.arange(12)#.reshape((3, 4))
print(a)
plt.hist(a)
plt.show()
#print(a)
#a = np.insert(a, 1, np.array([100, 101, 102]), axis=1)
#print(a)
#print(np.delete(a, 1, axis=1))
