import random
import time
import matplotlib.pyplot as plt


def merge_sort(array):
    if len(array) <= 1: return array

    mid = len(array) // 2
    left = merge_sort(array[:mid])
    right = merge_sort(array[mid:])

    return merge(left, right)


def merge(l, r):
    sorted_array = []
    i = j = 0

    while i < len(l) and j < len(r):
        if l[i] <= r[j]:
            sorted_array.append(l[i])
            i += 1
        else:
            sorted_array.append(r[j])
            j += 1

    sorted_array.extend(l[i:])
    sorted_array.extend(r[j:])

    return sorted_array


def array_generator(min_value=-1000, max_value=1000):
    arrays = []
    for size in range(1, 2000 + 1, 100):
        arrays.append([random.randint(min_value, max_value) for _ in range(1901)])
    return arrays


r_array = array_generator()
time_array = []
for arr in r_array:
    start_time = time.time()
    s_array = merge_sort(arr)
    end_time = time.time()
    time_array.append(end_time - start_time)


def plot(sizes, times):
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, times, marker='o', linestyle='-', color='b', label='Merge Sort Execution Time')
    plt.xlabel("Размер массива", fontsize=12)
    plt.ylabel("Время выполнения, сек", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


s = [i for i in range(1, 2000 + 1, 100)]
t = [0.0, 0.00010017156600952148, 0.00020126104354858398, 0.00030134916305541993, 0.0004527568817138672,
     0.000552511215209961, 0.0007045507431030273, 0.000853872299194336, 0.0010051965713500977, 0.001155245304107666,
     0.0012530684471130371, 0.0014577150344848634, 0.0016647696495056153, 0.0017620086669921874, 0.001857614517211914,
     0.0020629048347473146, 0.0022111773490905763, 0.0023120880126953126, 0.0024630069732666016, 0.0026106476783752442]

plot(s, t)