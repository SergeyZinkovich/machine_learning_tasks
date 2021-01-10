import numpy as np

n, m = map(int, input().split())

x = np.ndarray((n, m))
y = np.ndarray(n)

for i in range(n):
    a = list(map(float, input().split()))
    x[i] = a[:-1]
    y[i] = a[-1]

for i in range(n):
    ids = np.where(y != y[i])[0]
    yy_id = ids[np.argmin(np.linalg.norm(x[ids] - x[i], axis=1))]

    ids = np.where(y == y[i])[0]
    xx_id = ids[np.argmin(np.linalg.norm(x[ids] - x[yy_id], axis=1))]

    print(np.linalg.norm(x[xx_id] - x[yy_id]) / np.linalg.norm(x[i] - x[yy_id]))
