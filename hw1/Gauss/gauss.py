import numpy as np
import time
import matplotlib.pyplot as plt

def gauss(a, f, n):
    for k in range(n):
        a[k, k + 1 : n] = a[k, k + 1 : n] / a[k, k]
        f[k] = f[k] / a[k, k]
        a[k, k] = 1
        for i in range(k + 1, n):
            a[i, k + 1 : n] = a[i, k+1:n] - a[i, k] * a[k, k + 1:n]
            f[i] = f[i] - a[i, k] * f[k]
            a[i, k] = 0
    x = np.ones(n)
    for i in range(n - 1, -1, -1):
        x[i] = f[i]
        for j in range(i + 1, n):
            x[i] = x[i] - a[i, j] * x[j]
    return x

def gener_matrix(n, a):
    for i in range(n):
        tmp = 0
        for j in range(n):
            tmp += abs(a[i, j])
        a[i, i] = tmp
    return a

graph_time_sys = []
graph_size = []
graph_time = []

for i in range(50, 400, 50):
    a = np.random.rand(i, i)
    f = np.random.rand(i)
    a = gener_matrix(i, a)
    start = time.time()
    x = gauss(a, f, i)
    stop = time.time()
    graph_time.append(stop - start)
    graph_size.append(i)
    start = time.time()
    x1 = np.linalg.solve(a, f)
    stop = time.time()
    graph_time_sys.append(stop - start)
    assert(np.allclose(x, x1))
plt.xlabel('size of matrix')
plt.ylabel('time')
plt.title('Blue - my implementation, green - system implementation')
plt.plot(graph_size, graph_time)
plt.plot(graph_size, graph_time_sys)
plt.show()
