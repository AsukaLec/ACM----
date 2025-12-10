# def shell_sort_2(array):
#     n = len(array)
#     h = 1
#     while h < n / 3:
#         h = int(3 * h + 1)
#     while h >= 1:
#         for i in range(h, n):
#             j = i
#             while j >= h and array[j] < array[j - h]:
#                 array[j], array[j - h] = array[j - h], array[j]
#                 j -= h
#         h = int(h / 3)
#         for i in range(n):
#             print(array[i], end=' ')
#         print()

# if __name__ == "__main__":
#     arr = [49, 38, 65, 97, 76, 13, 27, 49, 55, 4, 62, 8, 2, 1, 3, 0]
#     for i in range(len(arr)):
#         print(arr[i], end=' ')
#     print()
#     shell_sort_2(arr)

N = 100010
w = n = 0
a = [0] * N
bucket = [[] for i in range(N)] 


def insertion_sort(A):
    for i in range(1, len(A)):
        key = A[i]
        j = i - 1
        while j >= 0 and A[j] > key:
            A[j + 1] = A[j]
            j -= 1
        A[j + 1] = key


def bucket_sort(array):
    bucket_size = int(w / n + 1)
    for i in range(0, n):
        bucket[i].clear()
    for i in range(1, n + 1):
        bucket[int(array[i] / bucket_size)].append(array[i])
    p = 0
    for i in range(0, n):
        insertion_sort(bucket[i])
        for j in range(0, len(bucket[i])):
            a[p] = bucket[i][j]
            p += 1

if __name__ == "__main__":
    n = int(input())
    w = 0
    for i in range(1, n + 1):
        array_i = int(input())
        a[i] = array_i
        if array_i > w:
            w = array_i
    bucket_sort(a)
    for i in range(0, n):
        print(a[i], end=' ')