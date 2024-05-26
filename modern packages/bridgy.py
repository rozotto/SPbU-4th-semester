def y1(x):
    return x * 2 - 3


def y2(x):
    return x ** 2


def distance(x):
    return y2(x) - y1(x)


def bridgy(y1, y2):
    minx = min([[distance(x), x] for x in range(-1000, 1001)])
    return minx[1], y1(minx[1]), minx[1], y2(minx[1])


result = bridgy(y1, y2)
print(f"Точки соединения мостом: ({result[0]}, {result[1]}) и ({result[2]}, {result[3]})")