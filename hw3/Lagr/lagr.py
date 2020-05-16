import numpy as np

def l(i, z):
    ans = 1
    for j in range(n):
        if i != j:
            ans = ans * (z - x[j])
            ans = ans / (x[i] - x[j])
    return ans

n = int(input())

x = list(map(float, input().split()))
x = np.array(x)
y = list(map(float, input().split()))
y = np.array(y)
m = int(input())
z = list(map(float, input().split()))
z = np.array(z)

file_ans = open("lagr_ans.txt", 'w')

for k in range(m):
    ans = 0
    for i in range(n):
        ans = ans + y[i] * l(i, z[k])
    file_ans.write(str(ans) + ' ')
