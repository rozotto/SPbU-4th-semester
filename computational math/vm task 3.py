import math
import numpy as np
from time import time


def F(x):
    return np.array([
        math.cos(x[1] * x[0]) - math.exp(-3 * x[2]) + x[3] * x[4] ** 2 - x[5] - math.sinh(2 * x[7]) * x[8] + 2 * x[9] + 2.000433974165385440,
        math.sin(x[1] * x[0]) + x[2] * x[8] * x[6] - math.exp(-x[9] + x[5]) + 3 * x[4] ** 2 - x[5] * (x[7] + 1) + 10.886272036407019994,
        x[0] - x[1] + x[2] - x[3] + x[4] - x[5] + x[6] - x[7] + x[8] - x[9] - 3.1361904761904761904,
        2 * math.cos(-x[8] + x[3]) + x[4] / (x[2] + x[0]) - math.sin(x[1] ** 2) + math.cos(x[6] * x[9]) ** 2 - x[7] - 0.1707472705022304757,
        math.sin(x[4]) + 2 * x[7] * (x[2] + x[0]) - math.exp(-x[6] * (-x[9] + x[5])) + 2 * math.cos(x[1]) - 1.0 / (-x[8] + x[3]) - 0.3685896273101277862,
        math.exp(x[0] - x[3] - x[8]) + x[4] ** 2 / x[7] + math.cos(3 * x[9] * x[1]) / 2 - x[5] * x[2] + 2.0491086016771875115,
        x[1] ** 3 * x[6] - math.sin(x[9] / x[4] + x[7]) + (x[0] - x[5]) * math.cos(x[3]) + x[2] - 0.7380430076202798014,
        x[4] * (x[0] - 2 * x[5]) ** 2 - 2 * math.sin(-x[8] + x[2]) + 0.15e1 * x[3] - math.exp(x[1] * x[6] + x[9]) + 3.5668321989693809040,
        7 / x[5] + math.exp(x[4] + x[3]) - 2 * x[1] * x[7] * x[9] * x[6] + 3 * x[8] - 3 * x[0] - 8.4394734508383257499,
        x[9] * x[0] + x[8] * x[1] - x[7] * x[2] + math.sin(x[3] + x[4] + x[5]) * x[6] - 0.78238095238095238096], dtype=float)


def J(x):
    return np.array(
        [[-x[1] * math.sin(x[1] * x[0]), -x[0] * math.sin(x[1] * x[0]), 3 * math.exp(-3 * x[2]), x[4] ** 2, 2 * x[3] * x[4],
          -1, 0, -2 * math.cosh(2 * x[7]) * x[8], -math.sinh(2 * x[7]), 2],
         [x[1] * math.cos(x[1] * x[0]), x[0] * math.cos(x[1] * x[0]), x[8] * x[6], 0, 6 * x[4],
          -math.exp(-x[9] + x[5]) - x[7] - 1, x[2] * x[8], -x[5], x[2] * x[6], math.exp(-x[9] + x[5])],
         [1, -1, 1, -1, 1, -1, 1, -1, 1, -1], [-x[4] / (x[2] + x[0]) ** 2, -2 * x[1] * math.cos(x[1] ** 2), -x[4] / (x[2] + x[0]) ** 2,
          -2 * math.sin(-x[8] + x[3]), 1.0 / (x[2] + x[0]), 0, -2 * math.cos(x[6] * x[9]) * x[9] * math.sin(x[6] * x[9]), -1,
          2 * math.sin(-x[8] + x[3]), -2 * math.cos(x[6] * x[9]) * x[6] * math.sin(x[6] * x[9])],
         [2 * x[7], -2 * math.sin(x[1]), 2 * x[7], 1.0 / (-x[8] + x[3]) ** 2, math.cos(x[4]),
          x[6] * math.exp(-x[6] * (-x[9] + x[5])), -(x[9] - x[5]) * math.exp(-x[6] * (-x[9] + x[5])), 2 * x[2] + 2 * x[0],
          -1.0 / (-x[8] + x[3]) ** 2, -x[6] * math.exp(-x[6] * (-x[9] + x[5]))],
         [math.exp(x[0] - x[3] - x[8]), -1.5 * x[9] * math.sin(3 * x[9] * x[1]), -x[5], -math.exp(x[0] - x[3] - x[8]),
          2 * x[4] / x[7], -x[2], 0, -x[4] ** 2 / x[7] ** 2, -math.exp(x[0] - x[3] - x[8]),
          -1.5 * x[1] * math.sin(3 * x[9] * x[1])], [math.cos(x[3]), 3 * x[1] ** 2 * x[6], 1, -(x[0] - x[5]) * math.sin(x[3]),
          x[9] / x[4] ** 2 * math.cos(x[9] / x[4] + x[7]),
          -math.cos(x[3]), x[1] ** 3, -math.cos(x[9] / x[4] + x[7]), 0, -1.0 / x[4] * math.cos(x[9] / x[4] + x[7])],
         [2 * x[4] * (x[0] - 2 * x[5]), -x[6] * math.exp(x[1] * x[6] + x[9]), -2 * math.cos(-x[8] + x[2]), 1.5,
          (x[0] - 2 * x[5]) ** 2, -4 * x[4] * (x[0] - 2 * x[5]), -x[1] * math.exp(x[1] * x[6] + x[9]), 0,
          2 * math.cos(-x[8] + x[2]), -math.exp(x[1] * x[6] + x[9])],
         [-3, -2 * x[7] * x[9] * x[6], 0, math.exp(x[4] + x[3]), math.exp(x[4] + x[3]),
          -7.0 / x[5] ** 2, -2 * x[1] * x[7] * x[9], -2 * x[1] * x[9] * x[6], 3, -2 * x[1] * x[7] * x[6]],
         [x[9], x[8], -x[7], math.cos(x[3] + x[4] + x[5]) * x[6], math.cos(x[3] + x[4] + x[5]) * x[6],
          math.cos(x[3] + x[4] + x[5]) * x[6], math.sin(x[3] + x[4] + x[5]), -x[2], x[1], x[0]]], dtype=float)


# функция-декоратор для подсчета времени
def time_check(func):
    def wrapper(*args, **kwargs):
        start = time()
        res = func(*args, **kwargs)
        end = time() - start
        return res, end
    return wrapper


x0 = np.array([0.5, 0.5, 1.5, -1.0, -0.5, 1.5, 0.5, -0.5, 1.5, -1.5])


# a)
@time_check
def Newton(x):
    eps = 1e-6
    iterations = 0
    operations = 0
    n = len(F(x))
    while True:
        iterations += 1
        operations += n ** 3 + n ** 2  # o(n^3) - решение СЛАУ, o(n^2) - умножение матрицы на вектор
        x_0 = x.copy()
        x -= np.linalg.inv(J(x)).dot(F(x))  # формула (1.2)
        if np.linalg.norm(x - x_0) < eps:  # замечание 1.1
            break
    return x, iterations, operations


print('Метод Ньютона')
tmp_res = Newton(x0.copy())
res, iterations, operations = tmp_res[0]
timing = tmp_res[1]
print(f'Значения корней: {res}')
print(f'Значения функции: {F(res)}')
print(f'Количество итераций {iterations} и операций {operations}')
print(f'Время расчета: {timing}')
print()


# b)
def LU_decomposition(matrix):
    n = len(matrix)
    U = matrix.copy()
    L = np.eye(n)
    P = np.eye(n)
    Q = np.eye(n)

    det_coef = 1
    for k in range(n - 1):
        i_piv = k
        j_piv = k
        for i in range(k, n):
            for j in range(k, n):
                if abs(U[i][j]) > abs(U[i_piv][j_piv]):
                    i_piv = i
                    j_piv = j

        if U[i_piv][j_piv] == 0:
            break

        if i_piv != k:
            U1 = U.copy()
            tmp = U1[k]
            U[k] = U1[i_piv]
            U[i_piv] = tmp

            P1 = P.copy()
            tmp = P1[k]
            P[k] = P1[i_piv]
            P[i_piv] = tmp

            det_coef *= -1

        if j_piv != k:
            U1 = U.copy()
            Q1 = Q.copy()
            for i in range(n):
                tmp = U1[i][k]
                U[i][k] = U1[i][j_piv]
                U[i][j_piv] = tmp

                tmp = Q1[i][k]
                Q[i][k] = Q1[i][j_piv]
                Q[i][j_piv] = tmp

                det_coef *= -1

        for i in range(k + 1, n):
            U[i][k] = U[i][k] / U[k][k]
            coef = -U[i][k]
            for j in range(k + 1, n):
                U[i][j] += U[k][j] * coef

    for i in range(1, n):
        for j in range(i):
            L[i][j] = U[i][j]
            U[i][j] = 0

    return L, U, P, Q


def solve_linear_system(A, b, L, U, P, Q):
    n = len(A)
    y = np.zeros(n)
    x = np.zeros(n)
    Pb = np.matmul(P, b)
    for i in range(n):
        y[i] = Pb[i] - sum(L[i, j] * y[j] for j in range(i))
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i + 1, n))) / U[i, i]
    x = np.matmul(Q, x)
    return x


@time_check
def Modified_Newton(x):
    eps = 1e-6
    iterations = 0
    operations = 0
    n = len(F(x))
    L, U, P, Q = LU_decomposition(J(x))  # замечание 1.3, где считаем матрицу лишь в начальной точке
    while True:
        iterations += 1
        operations += 2 * n ** 2 + 2 * n  # o(2n^2+2n) - решение СЛАУ через LU
        x_0 = x.copy()
        x += np.array([*solve_linear_system(J(x), -F(x), L, U, P, Q)])  # замечание 1.3
        if np.linalg.norm(x - x_0) < eps:  # замечание 1.1
            break
    return x, iterations, operations


print('Модифицированный метод Ньютона')
tmp_res = Modified_Newton(x0.copy())
res, iterations, operations = tmp_res[0]
timing = tmp_res[1]
print(f'Значения корней: {res}')
print(f'Значения функции: {F(res)}')
print(f'Количество итераций {iterations} и операций {operations}')
print(f'Время расчета: {timing}')
print()


# c)
@time_check
def k_Newton(x):
    k = 5
    eps = 1e-6
    iterations = 0
    operations = 0
    n = len(F(x))
    L, U, P, Q = LU_decomposition(J(x))
    while True:
        iterations += 1
        operations += 2 * n ** 2 + 2 * n
        x_0 = x.copy()
        x += np.array([*solve_linear_system(J(x), -F(x), L, U, P, Q)])  # замечание 1.3
        if np.linalg.norm(x - x_0) < eps:  # замечание 1.1
            break
        if iterations < k:  # пока количество итераций меньше начального k будем пересчитывать матрицу
            L, U, P, Q = LU_decomposition(J(x))
            operations += n ** 3  # o(n^3) - решение СЛАУ
    return x, iterations, operations


print('Метод Ньютона с переходом к модифицированному')
tmp_res = k_Newton(x0.copy())
res, iterations, operations = tmp_res[0]
timing = tmp_res[1]
print(f'Значения корней: {res}')
print(f'Значения функции: {F(res)}')
print(f'Количество итераций {iterations} и операций {operations}')
print(f'Время расчета: {timing}')
print()


# d)
@time_check
def m_Newton(x):
    m = 5
    eps = 1e-6
    iterations = 0
    operations = 0
    n = len(F(x))
    L, U, P, Q = LU_decomposition(J(x))
    while True:
        iterations += 1
        operations += 2 * n ** 2 + 2 * n
        x_0 = x.copy()
        x += np.array([*solve_linear_system(J(x), -F(x), L, U, P, Q)])  # замечание 1.3
        if np.linalg.norm(x - x_0) < eps:  # замечание 1.1
            break
        if iterations % m == 0:  # каждые m итераций == итерация кратна m
            L, U, P, Q = LU_decomposition(J(x))
            operations += n ** 3
    return x, iterations, operations


print('Метод Ньютона с пересчетом матрицы Якоби каждые m итераций')
tmp_res = m_Newton(x0.copy())
res, iterations, operations = tmp_res[0]
timing = tmp_res[1]
print(f'Значения корней: {res}')
print(f'Значения функции: {F(res)}')
print(f'Количество итераций {iterations} и операций {operations}')
print(f'Время расчета: {timing}')
print()


# e)
@time_check
def k_m_Newton(x):
    k = 5
    m = 5
    eps = 1e-6
    iterations = 0
    operations = 0
    n = len(F(x))
    L, U, P, Q = LU_decomposition(J(x))
    while True:
        iterations += 1
        operations += 2 * n ** 2 + 2 * n
        x_0 = x.copy()
        x += np.array([*solve_linear_system(J(x), -F(x), L, U, P, Q)])  # замечание 1.3
        if np.linalg.norm(x - x_0) < eps:  # замечание 1.1
            break
        if iterations % m == 0 or iterations < k:
            L, U, P, Q = LU_decomposition(J(x))
            operations += n ** 3
    return x, iterations, operations


print('Метод Ньютона с m и k')
tmp_res = k_m_Newton(x0.copy())
res, iterations, operations = tmp_res[0]
timing = tmp_res[1]
print(f'Значения корней: {res}')
print(f'Значения функции: {F(res)}')
print(f'Количество итераций {iterations} и операций {operations}')
print(f'Время расчета: {timing}')
print()

# f)
x0_f = x0.copy()
x0_f[4] = -0.2
print(k_Newton(x0_f.copy()))
