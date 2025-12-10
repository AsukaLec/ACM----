import random

def merge_sort(arr):   
    def _merge_sort(arr, ll, rr):
        if rr - ll <= 1:
            return
        # 分解
        mid = (rr + ll) // 2
        _merge_sort(arr, ll, mid)
        _merge_sort(arr, mid, rr)
        # 合并
        def merge(a, b):
            i, j = 0, 0
            c = []
            while i < len(a) and j < len(b):
                # <!> 先判断 b[j] < a[i]，保证稳定性
                if b[j] < a[i]:
                    c.append(b[j])
                    j += 1
                else:
                    c.append(a[i])
                    i += 1
            # 此时一个数组已空，另一个数组非空，将非空的数组并入 c 中
            c.extend(a[i:])
            c.extend(b[j:])
            return c
        arr[ll:rr] = merge(arr[ll:mid], arr[mid:rr])
    _merge_sort(arr, 0,len(arr))


def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr)//2
        L, R = arr[:mid], arr[mid:]
        merge_sort(L); merge_sort(R)
        #此时 [0, mid], [mid + 1, len(arr) - 1] 已经排好顺序了
        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]; i += 1
            else:
                arr[k] = R[j]; j += 1
            k += 1
        while i < len(L):
            arr[k] = L[i]; i += 1; k += 1
        while j < len(R):
            arr[k] = R[j]; j += 1; k += 1
       
       
arr = list(range(17))
random.shuffle(arr)
# arr = [7, 5, 4, 11, 12, 3, 2]

for i in range (len(arr)):
    print (arr[i], end=" ")

print()


merge_sort(arr)
for i in range(len(arr)):
    print(arr[i],end=" ")