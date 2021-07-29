import numpy as np
a = np.zeros((8, 8), dtype=np.int)
a[0:3, 0:8] = 1
a[5:9, 0:8] = 1

a[0:8, 0:3] = 1
a[0:8, 6:8] = 1
print(a)
