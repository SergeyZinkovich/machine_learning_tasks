import numpy as np

n, dim, k, t = map(int, input().split())

x, y = [], []

for _ in range(n):
    a = input().split()
    y.append(int(a[-1]))
    x.append(list(map(float, a[:-1])))

x = np.array(x)

for __ in range(t):
    m = np.zeros(shape=(k, dim))
    c = [0 for _ in range(k)]

    for i in range(n):
        m[y[i]] += x[i]
        c[y[i]] += 1

    for i in range(k):
        m[i] /= c[i]

    y_new = y[:]
    for i in range(n):
        _class = 0
        _norm = float('inf')
        for j in range(k):
            norm = np.linalg.norm(m[j] - x[i])
            if norm < _norm:
                _class = j
                _norm = norm
        y_new[i] = _class

    if y == y_new:
        break

    y = y_new

print(*y, sep='\n')
