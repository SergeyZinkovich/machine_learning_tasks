import numpy as np
import sys

n, m, k = map(int, input().split())

h = np.full(k, 1 / k)
count = np.full(k, 1)

p = np.full((k, m), 1 / m)
c = np.full((k, m), 1)
# c_sums = np.full(k, m)

a = np.array(list(map(lambda x: list(map(int, x.split())), sys.stdin.readlines())))

for i in range(n):
    x = a[i, :-1]
    y = a[i][-1]

    # h_1 = h.copy()
    # for ii in range(len(a[:-1])):
    #     h_1 *= p[:, ii] if a[ii] == 1 else 1
    temp = x * p
    temp[temp == 0] = 1
    temp = np.prod(temp, axis=1)
    print(np.argmax(h * temp))

    c[y] += x
    # c_sums[a[-1]] += sum(a[:-1])
    p[y] = (c[y]) / (c[y].sum())
    # p[a[-1]] = (c[a[-1]]) / c_sums[a[-1]]

    count[y] += 1
    h = count / (i + k)





