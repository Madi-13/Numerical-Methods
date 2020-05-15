import numpy as np
import scipy.linalg as sl
from math import sqrt
import time
import matplotlib.pyplot as plt

def matrix_s(n, a):
    s = np.ones((n, n)) * 0
    for i in range(n):
        tmp = 0
        for k in range(i - 1):
            tmp = tmp + s[k, i] * s[k + 1, i]
        s[i, i] = sqrt((a[i, i] - tmp))
    for i in range(n):
        for j in range(i, n):
            tmp = np.sum(s[:i, i] * s[:i, j])
            s[i, j] = (a[i, j] - tmp) / s[i, i]
    return s

def gener_matrix(n, a):
    for i in range(n):
        tmp = 0
        for j in range(n):
            tmp += abs(a[i, j])
        a[i, i] = tmp
    for i in range(n):
        for j in range(n):
            a[i, j] = a[j, i]
    return a

graph_size = []
graph_time = []
graph_time_sys = []

for size in range(100, 350, 50):
    a = np.random.rand(size, size)
    f = np.random.rand(size)
    a = gener_matrix(size, a)
    start = time.time()
    s = matrix_s(size, a)
    st = s.transpose()
    y = np.linalg.solve(st, f)
    x = np.linalg.solve(s, y)
    stop = time.time()
    graph_time.append(stop - start)
    graph_size.append(size)
    #system solve
    start = time.time()
    s1 = sl.cholesky(a)
    st = s1.transpose()
    y = np.linalg.solve(st, f)
    x1 = np.linalg.solve(s, y)
    stop = time.time()
    graph_time_sys.append(stop - start)
plt.xlabel('size of matrix')
plt.ylabel('time')
plt.title('Blue - my implementation, green - system implementation')
plt.plot(graph_size, graph_time)
plt.plot(graph_size, graph_time_sys)
plt.show()
