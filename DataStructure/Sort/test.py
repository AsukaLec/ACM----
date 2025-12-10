import sys
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


# ------------------ 排序算法集合 ------------------
#冒泡排序
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True
        if not swapped:
            break

#选择排序
def selection_sort(arr):
    for i in range(len(arr)-1):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

#锦标赛排序
INF = sys.maxsize
def tournament_sort(arr):
    n = len(arr)
    if n <= 1:
        return arr.copy()

    tmp = [0] * (n * 2)          # 树节点
    out = [0] * n                # 结果数组

    def winner(pos1, pos2):
        # 叶节点直接取 idx，内部节点取 tmp 里存的 idx
        u = pos1 if pos1 >= n else tmp[pos1]
        v = pos2 if pos2 >= n else tmp[pos2]
        # 稳定关键：<= 保证左边先出
        return u if tmp[u] <= tmp[v] else v

    #建立锦标赛树
    for i in range(n):
        tmp[n + i] = arr[i]      # 叶节点存真实值
    for i in range(2 * n - 1, 1, -2):
        k = i // 2
        tmp[k] = winner(i, i - 1)
    
    # for i in range(2*n):
    #     print(tmp[i])

    #依次选出最小值
    for i in range(n):
        root = tmp[1]            # 当前全局最小所在的叶 idx
        out[i] = tmp[root]   # 写入结果
        tmp[root] = INF      # 标记为“已删除”
        # 沿路径向上重建
        pos = root
        while pos > 1:
            par = pos // 2
            sibling = pos + 1 if pos % 2 == 0 else pos - 1
            tmp[par] = winner(pos, sibling)
            pos = par

    arr[:] = out
    # for i in range(n):
    #     print(out[i])

#堆排序
def heap_sort(arr):
    n = len(arr)
    
    def sift_down(arr, start, end):
        # 计算父结点和子结点的下标
        parent = int(start)
        child = int(parent * 2 + 1)
        while child <= end:  # 子结点下标在范围内才做比较
            # 先比较两个子结点大小，选择最大的
            if child + 1 <= end and arr[child] < arr[child + 1]:
                child += 1
            # 如果父结点比子结点大，代表调整完毕，直接跳出函数
            if arr[parent] >= arr[child]:
                return
            else:  # 否则交换父子内容，子结点再和孙结点比较
                arr[parent], arr[child] = arr[child], arr[parent]
                parent = child
                child = int(parent * 2 + 1)

    # 从最后一个节点的父节点开始 sift down 以完成堆化 (heapify)
    i = (n - 1 - 1) / 2
    while i >= 0:
        sift_down(arr, i, n - 1)
        i -= 1
    # 先将第一个元素和已经排好的元素前一位做交换，再重新调整（刚调整的元素之前的元素），直到排序完毕
    i = n - 1
    while i > 0:
        arr[0], arr[i] = arr[i], arr[0]
        sift_down(arr, 0, i - 1)
        i -= 1

#插入排序
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

#折半插入排序
def binary_insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        # 在已排序区间 [0, i-1] 中二分查找插入位置
        left, right = 0, i
        while left < right:
            mid = (left + right) // 2
            if arr[mid] > key:
                right = mid
            else:
                left = mid + 1
        # 把 [left, i-1] 整体右移一位
        for j in range(i, left, -1):
            arr[j] = arr[j - 1]
        arr[left] = key

#快速排序
def quick_sort(arr):
    def _q(a, l, r):
        if l >= r: return
        pivot = a[(l+r)//2]
        i, j = l, r
        while i <= j:
            while a[i] < pivot: i += 1
            while a[j] > pivot: j -= 1
            if i <= j:
                a[i], a[j] = a[j], a[i]
                i += 1; j -= 1
        _q(a, l, j); _q(a, i, r)
    _q(arr, 0, len(arr)-1)

#归并排序
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

def built_in_sort(arr):
    arr.sort()

# ------------------ 基准测试引擎 ------------------
ALGORITHMS = {
    "Selection": selection_sort,    
    "Tournament": tournament_sort,
    "Heap": heap_sort,
    "Insertion": insertion_sort,
    "BinInsertion":binary_insertion_sort,
    "Bubble": bubble_sort,
    "Quick": quick_sort,    
    "Merge": merge_sort,
    "Python TimSort": built_in_sort
}

def run_single(alg, data):
    """返回一次排序耗时（秒）"""
    arr = data.copy()
    t0 = time.perf_counter()
    alg(arr)
    t1 = time.perf_counter()
    assert arr == sorted(data)  # 简单正确性检查
    return t1 - t0

def benchmark(sizes, repeat=3):
    """对每一规模跑 repeat 次取平均"""
    results = {name: [] for name in ALGORITHMS}
    for n in sizes:
        print(f"testing n={n} ...")
        data = list(range(n))
        random.shuffle(data)
        for name, func in ALGORITHMS.items():
            times = [run_single(func, data) for _ in range(repeat)]
            results[name].append(np.mean(times))
    return pd.DataFrame(results, index=pd.Index(sizes, name='Size'))

# ------------------ 可视化 ------------------
def plot_results(df, loglog=True, save_path=None):
    plt.figure(figsize=(7, 5))
    for col in df.columns:
        plt.plot(df.index, df[col], marker='o', label=col)
    if loglog:
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel("Array size n (log scale)")
        plt.ylabel("Time / s (log scale)")
    else:
        plt.xlabel("Array size n")
        plt.ylabel("Time / s")
    plt.title("Sorting Algorithm Time Complexity")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

# ------------------ 主入口 ------------------
if __name__ == "__main__":
    # 1. 数据规模（可按需改）
    sizes = [100, 500, 1000, 2000, 5000, 10000, 20000]
    # 2. 跑基准
    df = benchmark(sizes, repeat = 3)
    print(df)
    # 3. 绘图
    plot_results(df, loglog=False, save_path="sort_complexity.png")
    plot_results(df, loglog=True, save_path="sort_complexity_loglog.png")
