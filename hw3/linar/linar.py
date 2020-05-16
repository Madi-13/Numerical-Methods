import numpy as np

n = int(input())

x = list(map(float, input().split()))
x = np.array(x)
y = list(map(float, input().split()))
y = np.array(y)
m = int(input())
z = list(map(float, input().split()))
z = np.array(z)

file_ans = open('linar_ans.txt', 'w')

for i in range(m):
    j = 1
    while not (z[i] <= x[j] and z[i] >= x[j - 1]):
        j = j + 1
    ans = (z[i] - x[j - 1]) / (x[j] - x[j - 1]) * (y[j] - y[j - 1]) + y[j - 1]
    file_ans.write(str(ans) + ' ')
