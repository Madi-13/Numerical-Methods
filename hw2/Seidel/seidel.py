import numpy as np
import time
import matplotlib.pyplot as plt

eps = 0.2

def diff(x, xnew, n):
    max_val = 0
    for i in range(n):
        if abs(x[i] - xnew[i]) > max_val:
            max_val = abs(x[i] - xnew[i])
    return max_val

def seidel(A, f, x, n):
    xnew = np.ones(n) * 0
    for i in range(n):
        s = 0
        for j in range(i):
            s = s + A[i][j] * xnew[j]
        for j in range(i + 1, n):
            s = s + A[i][j] * x[j]
        xnew[i] = (f[i] - s) / A[i][i]
    return xnew
    
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

for i in range(50, 700, 50):
    a = np.random.rand(i, i)
    f = np.random.rand(i)
    a = gener_matrix(i, a)
    x = np.ones(i) * 0
    start = time.time()
    xnew = seidel(a, f, x, i)
    while diff(x, xnew, i) > eps:
        x = xnew
        xnew = seidel(a, f, x, i)
    stop = time.time()
    graph_time.append(stop - start)
    graph_size.append(i)
    start = time.time()
    x1 = np.linalg.solve(a, f)
    stop = time.time()
    graph_time_sys.append(stop - start)
plt.xlabel('size of matrix')
plt.ylabel('time')
plt.title('Blue - seidel, green - linalg.solve')
plt.plot(graph_size, graph_time)
plt.plot(graph_size, graph_time_sys)
plt.show()
