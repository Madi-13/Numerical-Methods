import numpy as np
import scipy.linalg as sl
import time
import matplotlib.pyplot as plt

def sweep(n, matrix, f, a, b, c):
    x = np.ones(n + 1) * 0
    for i in range(1, n + 1):
        a[i] = matrix[i, i - 1]
        b[i] = matrix[i, i]
        c[i] = matrix[i, i + 1]
    alpha = [0] * (n + 2)
    beta = [0] * (n + 2)
    x = [0] * (n + 2)
    alpha[1] = 0
    beta[1] = 0
    for i in range(1, n + 1):
        d = a[i] * alpha[i] + b[i]
        alpha[i + 1] = - c[i] / d
        beta[i + 1] = (f[i] - a[i] * beta[i]) / d
    x[n + 1] = 0
    for i in range(n, 0, -1):
        x[i] = alpha[i + 1] * x[i + 1] + beta[i + 1]
    return x[1:n + 1]

def gener_matrix(n, a):        
    for i in range(1, n + 1):
        tmp = 0
        for j in range(1, n + 1):
            tmp += abs(a[i][j])
        a[i][i] = tmp
    for i in range(1, n + 1):
        j = 2
        while i + j <= n:
            a[i][i + j] = 0
            j = j + 1
        j = 2
        while i - j >= 1:
            a[i][i - j] = 0
            j = j + 1
    return a

graph_size = []
graph_time = []
graph_size_sys = []
graph_time_sys = []

for size in range(1000, 3000, 500):
    matrix = np.random.rand(size + 2, size + 2)
    f = np.random.rand(size + 2)
    abc = np.ones((3, size)) * 0
    matrix = gener_matrix(size, matrix)
    a = np.ones(size + 1) * 0
    b = np.ones(size + 1) * 0
    c = np.ones(size + 1) * 0
    start = time.time()
    x = sweep(size, matrix, f, a, b, c)
    stop = time.time()
    graph_time.append(stop - start)
    graph_size.append(size)
    abc[0] = a[1:size + 1]
    abc[1] = b[1:size + 1]
    abc[2] = c[1:size + 1]
    f1 = f[1 : size + 1]
    start = time.time()
    x1 = sl.solve_banded((1, 1), abc, f1)
    stop = time.time()
    graph_time_sys.append(stop - start)
plt.xlabel('size of matrix')
plt.ylabel('time')
plt.title('Blue - my implementation, green - system implementation')
plt.plot(graph_size, graph_time)
plt.plot(graph_size, graph_time_sys)
plt.show()
