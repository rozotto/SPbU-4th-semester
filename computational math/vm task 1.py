from math import cosh, sin, sqrt, factorial
from decimal import Decimal
from pandas import DataFrame


def z(x):
    return Decimal(cosh(sqrt(x ** 2 + 0.3) / (1 + x)) * sin((1 + x) / (0.6 * x)))


def ch(x):
    res = 0
    eps = 10 ** (-6) / 1.26
    x_n = 1.0
    k = 0
    x_last = 0
    while abs(x_n) > eps:
        x_last = res
        x_n = Decimal((x ** (2 * k))) / Decimal(factorial(2 * k))
        res += x_n
        k += 1
    return x_last


def sinn(x):
    res = 0
    eps = 10 ** (-6) / 3.36
    x_n = 1.0
    k = 0
    x_last = 0
    while abs(x_n) > eps:
        x_last = res
        x_n = Decimal((-1) ** k * (x ** (2 * k + 1))) / Decimal(factorial(2 * k + 1))
        res += x_n
        k += 1
    return x_last


def koren(x):
    eps = 10 ** (-6) / 0.54
    a = float(x)
    x0 = 1.0
    x1 = a
    x_last = 0
    while abs(x0 - x1) > eps:
        x_last = x1
        x0 = x1
        x1 = 0.5 * (x1 + a / x1)
    return x_last


x = [i / 100 for i in range(20, 30 + 1)]
f_exact = [z(i) for i in x]
f_approx = [ch(koren(i ** 2 + 0.3) / (1 + i)) * sinn((1 + i) / (0.6 * i)) for i in x]
error = [abs(f_exact[i] - f_approx[i]) for i in range(len(f_exact))]
DataFrame({'x': x, 'z(x)': f_exact, 'z_(x)': f_approx, '|dz(x)|': error}).to_excel('res.xlsx', index=False)
