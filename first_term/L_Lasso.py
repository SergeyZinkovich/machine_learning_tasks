from sklearn import linear_model
import numpy as np


nn = int(input())

for _ in range(nn):
    coefs = input().split()
    n, m, k = map(int, coefs[:-1])
    s = float(coefs[-1])
    x, y = [], []
    for __ in range(n):
        a = list(map(float, input().split()))
        y.append(a[-1])
        x.append(a[:-1])
    x = np.array(x)
    y = np.array(y)
    model = linear_model.Lasso(alpha=1.9).fit(x, y)
    print()
    print(*model.coef_, model.intercept_)

    print(model.score(x, y), s, model.score(x, y) > s)
    print(m - len(model.coef_.nonzero()[0]), k, m - len(model.coef_.nonzero()[0]) == k)
