import numpy as np
import math


# 1) LU-разложение матрицы A
def LU_decomposition(A):
    n = len(A)
    L = np.eye(n)
    U = np.copy(A)
    P = np.eye(n)
    Q = np.eye(n)

    for k in range(n - 1):
        max_index = np.argmax(abs(U[k:, k])) + k
        if max_index != k:
            U[[k, max_index]] = U[[max_index, k]]
            L[[k, max_index], :k] = L[[max_index, k], :k]
            P[[k, max_index]] = P[[max_index, k]]
        for j in range(k + 1, n):
            L[j, k] = U[j, k] / U[k, k]
            U[j, k:] -= L[j, k] * U[k, k:]
    return L, U, P, Q


res = []
for i in range(1000):
    size = np.random.randint(2, 10)
    A = np.array(np.random.randint(-256, 256, (size, size)), dtype=float)
    L, U, P, Q = LU_decomposition(A)
    LU = np.matmul(L, U)
    PA = np.matmul(P, A)
    res.append(np.allclose(LU, PA))
print(np.all(res))


# a) Определитель матрицы A
def detLU(L, U, P):
    det = np.linalg.det(P)
    for i in range(len(L)):
        det *= L[i, i] * U[i, i]
    return det


res = []
for i in range(1000):
    size = np.random.randint(2, 10)
    A = np.array(np.random.randint(-256, 256, (size, size)), dtype=float)
    L, U, P, Q = LU_decomposition(A)
    detA = detLU(L, U, P)
    res.append(np.allclose(detA, np.linalg.det(A)))
print(np.all(res))


# b) Решение СЛАУ Ax = b, выполнить проверку равенства Ax − b = 0
def solve_linear_system(A, b):
    L, U, P, Q = LU_decomposition(A)
    n = len(A)
    y = np.zeros(n)
    x = np.zeros(n)
    if detLU(L, U, P) == 0.0: return np.zeros(n)
    Pb = np.matmul(P, b)
    for i in range(n):
        y[i] = Pb[i] - sum(L[i, j] * y[j] for j in range(i))
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i + 1, n))) / U[i, i]
    return x


res = []
for i in range(1000):
    size = np.random.randint(2, 10)
    A = np.array(np.random.randint(-256, 256, (size, size)), dtype=float)
    b = np.matmul(A, np.array(np.random.randint(-256, 256, size), dtype=float))
    x = solve_linear_system(A, b)
    res.append(np.allclose(x, np.linalg.solve(A, b)))
print(np.all(res))


# c) Матрицу A−1 (выполнить проверку AA−1 и A−1A)
def inverse(A):
    n = len(A)
    invA = np.zeros_like(A, dtype=float)
    for i in range(n):
        b = np.zeros(n)
        b[i] = 1
        invA[:, i] = solve_linear_system(A, b)
    return invA


res = []
for i in range(1000):
    size = np.random.randint(2, 10)
    A = np.array(np.random.randint(-256, 256, (size, size)), dtype=float)
    invA = inverse(A)
    A_invA = np.matmul(A, invA)
    invA_A = np.matmul(invA, A)
    res.append(np.allclose(A_invA, invA_A))
print(np.all(res))


# d) Число обусловленности матрицы A
def condition_number(A):
    return np.linalg.norm(A, np.inf) * np.linalg.norm(inverse(A), np.inf)


res = []
for i in range(1000):
    size = np.random.randint(2, 10)
    A = np.array(np.random.randint(-256, 256, (size, size)), dtype=float)
    res.append(np.allclose(condition_number(A), np.linalg.cond(A, p=np.inf)))
print(np.all(res))


# 2) Нахождение ранга вырожденных матриц, проверка на совместность, частное решение
# Ранг матрицы равен количеству ненулевых диагональных элементов в верхнетреугольной матрице U из LU-разложения.
def LU(matrix):
    n = len(matrix)
    U = matrix.copy()
    L = np.eye(n)

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

            det_coef *= -1

        if j_piv != k:
            U1 = U.copy()
            for i in range(n):
                tmp = U1[i][k]
                U[i][k] = U1[i][j_piv]
                U[i][j_piv] = tmp
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

    return L, U


def rank(A):
    _, U = LU(A)
    r = np.sum(np.abs(np.diag(U)) > 0)
    return r


res = []
np.seterr(divide='ignore', invalid='ignore')
for _ in range(1000):
    size = np.random.randint(2, 10)
    A = np.array(np.random.randint(-256, 256, (size, size)), dtype=np.float64)
    singular_A = np.random.randint(2, size + 1)
    for i in range(singular_A):
        A[:, i] = A[:, 0]  # дублируем столбцы
        # A[i, :] = A[0, :]  # дублируем строки
    res.append(np.allclose(np.linalg.matrix_rank(A), rank(A)))
print(np.all(res))


# Теорема Кронекера—Капелли (критерий совместности системы линейных алгебраических уравнений)
def gauss_method(Ab):
    rows, cols = Ab.shape
    for i in range(min(rows, cols - 1)):
        Ab[i, :-1] = Ab[i, :-1] / Ab[i, i]
        for j in range(i - 1, -1, -1):
            koef = Ab[j, i]
            Ab[j, :-1] -= koef * Ab[i, :-1]
            Ab[j, -1] -= koef * Ab[i, -1]
    return Ab


size = np.random.randint(2, 4)
A = np.array(np.random.randint(-10, 10, (size, size)), dtype=np.float64)
singular_A = np.random.randint(2, size + 1)
for i in range(singular_A):
    A[i, :] = A[0, :]

# метод Гаусса
for i in range(size):
    if A[i, i] == 0:
        for j in range(i + 1, size):
            if A[j, i] != 0:
                A[[i, j]] = A[[j, i]]
                break
        else:
            continue

    for j in range(i + 1, size):
        koef = A[j, i] / A[i, i]
        A[j, i:] -= koef * A[i, i:]
A = A[:rank(A)]

b = np.matmul(A, np.array(np.random.randint(-5, 5, size), dtype=float))
Ab = np.hstack([A, b.reshape(-1, 1)])

# Частное решение
if rank(A) == rank(Ab):
    x = gauss_method(Ab)
    print(f'Общее решение: {x[:, :-1]}')
    print(f'Частное решение: {x[:, -1]}')
else:
    print('Система несовместна')


# 3) Реализовать QR-разложение матрицы
def QR_decomposition(A):
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()

    for j in range(n):
        v = R[j:, j].copy()  # Берем столбец матрицы R начиная с j-го элемента
        v[0] += np.copysign(np.linalg.norm(v), v[0])  # Добавляем норму столбца с учетом знака первого элемента
        v = v / np.linalg.norm(v)  # Нормируем вектор

        Q_j = np.eye(m)  # Создаем единичную матрицу размера m
        Q_j[j:, j:] -= 2 * np.outer(v, v)  # Создаем матрицу Хаусхолдера
        Q = np.dot(Q, Q_j.T)  # Умножаем Q на транспонированную матрицу Хаусхолдера
        R = np.dot(Q_j, R)  # Умножаем матрицу Хаусхолдера на R
    return Q, R


res = []
for i in range(1000):
    size = np.random.randint(2, 10)
    A = np.array(np.random.randint(-256, 256, (size, size)), dtype=float)
    Q1, R1 = QR_decomposition(A)
    Q2, R2 = np.linalg.qr(A)
    res.append(np.allclose(np.matmul(Q1, R1), np.matmul(Q2, R2)))
print(np.all(res))


# Решение невырожденной СЛАУ Ax=b
def solve_linear_qr(A, b):
    Q, R = QR_decomposition(A)
    n = len(A)
    y = np.matmul(np.transpose(Q), b)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(R[i, j] * x[j] for j in range(i + 1, n))) / R[i, i]
    return x


res = []
for i in range(1000):
    size = np.random.randint(2, 10)
    A = np.array(np.random.randint(-256, 256, (size, size)), dtype=float)
    b = np.matmul(A, np.array(np.random.randint(-256, 256, size), dtype=float))
    res.append(np.allclose(solve_linear_qr(A, b), np.linalg.solve(A, b)))
print(np.all(res))

# 4) метод Якоби и метод Зейделя решения СЛАУ


# создаем рандомную матрицу и приводим к матрице с диагональным преобладанием
def diagonally_dominant_matrix(A):
    for i in range(len(A)):
        A[i, i] = np.sum(np.abs(A[:, i])) * np.random.randint(2, 10) * np.random.choice([1, -1])
    return A


size = 3
A = np.array(np.random.randint(-256, 256, (size, size)), dtype=float)
A_diag = diagonally_dominant_matrix(A.copy())
b = np.matmul(A_diag, np.array(np.random.randint(-256, 256, size), dtype=float))


# метод простых итераций, где B = -D^(-1)(L+R), c = D^(-1)b
def B_c(A, b):
    B = np.matmul(-np.linalg.inv(np.diag(np.diag(A.copy()))), (np.triu(A.copy(), 1) + np.tril(A.copy(), -1)))
    c = np.matmul(np.linalg.inv(np.diag(np.diag(A.copy()))), b.copy())

    return B, c


# получим q - знаменатель геометрической прогрессии (нужен для сходимости) из условия ||B|| = q < 1
def qq(B):
    normB = np.linalg.norm(B, np.inf)
    if normB < 1:
        q = normB
    if normB >= 1:
        q = 0.99
    return q


B, c = B_c(A_diag, b)
q = qq(B)

# априорная оценка q^k / (1 - q) * ||c|| = eps
eps = 1e-10
k = int(math.log(((eps / np.linalg.norm(c, np.inf)) * (1 - q)), q))


# метод Якоби
def Jacobi(A, b, eps):
    B, c = B_c(A, b)
    q = qq(B)
    cnt = 1
    x_Jacobi = np.zeros_like(c)
    norm = np.linalg.norm(c, np.inf)

    while (q / (1 - q)) * norm > eps:
        x = x_Jacobi.copy()
        x_Jacobi = np.dot(B, x_Jacobi) + c
        diff = x_Jacobi - x
        norm = np.linalg.norm(diff, np.inf)
        cnt += 1

    return x_Jacobi, cnt


# метод Зейделя
def Seidel(A, b, eps, k):
    n = len(A)
    cnt = 1
    x_Seidel = np.zeros_like(b)
    for i in range(k):
        cnt += 1
        for i in range(n):
            sum1 = sum(A[i, j] * x_Seidel[j] for j in range(i))
            sum2 = sum(A[i, j] * x_Seidel[j] for j in range(i + 1, n))
            x_Seidel[i] = (b[i] - sum1 - sum2) / A[i, i]
        if np.linalg.norm(A.dot(x_Seidel) - b, ord=np.inf) <= eps:
            break
    return x_Seidel, cnt


print('МАТРИЦА С ДИАГОНАЛЬНЫМ ПРЕОБЛАДАНИЕМ')
print('Априорная оценка:', k)
answerJ, k_Jacobi = Jacobi(A_diag, b, eps)
answerS, k_Seidel = Seidel(A_diag, b, eps, k)
print(f'Решение методом Якоби: {answerJ}, апостериорная оценка: {k_Jacobi}')
print(f'Решение методом Зейделя: {answerS}, апостериорная оценка: {k_Seidel}')
print('Решение через встроенную функцию:', np.linalg.solve(A_diag, b))
print()


# Матрица положительно определена тогда и только тогда, когда все её собственные значения положительны.
def positive_and_diagonal(A):
    n = len(A)
    for i in range(n):
        sum = 0
        for j in range(n):
            if i != j:
                sum += abs(A[i][j])
        if abs(A[i][i]) <= abs(sum):
            return False
        return np.all(np.linalg.eigvals(A) > 0)


size = 3
#A1 = np.array(np.random.randint(-256, 256, (size, size)), dtype=float)
A1 = np.array([[ 232., -115., 81.], [-245., 62., 22.], [93., -214., 136.]], dtype=float)
b1 = np.matmul(A1, np.array(np.random.randint(-256, 256, size), dtype=float))
if positive_and_diagonal(A1):
    B1, c1 = B_c(A1, b1)
    q1 = qq(B1)
    eps = 1e-10
    k1 = int(math.log(((eps / np.linalg.norm(c1, np.inf)) * (1 - q1)), q1))
    print('МАТРИЦА ПОЛОЖИТЕЛЬНО ОПРЕДЕЛЕНА И БЕЗ ДИАГОНАЛЬНОГО ПРЕОБЛАДАНИЯ')
    print('Априорная оценка:', k1)
    answerJ, k_Jacobi = Jacobi(A1, b1, eps)
    answerS, k_Seidel = Seidel(A1, b1, eps, k1)
    print(f'Решение методом Якоби: {answerJ}, апостериорная оценка: {k_Jacobi}')
    print(f'Решение методом Зейделя: {answerS}, апостериорная оценка: {k_Seidel}')
    print('Решение через встроенную функцию:', np.linalg.solve(A1, b1))