[TOC]

# <center> 排序算法们~
## 零、综述
<div style="display:flex; gap:2em; justify-content:center;">
  <div style="text-align:center;">
    <img src="排序算法1.png" width="600"/>
  </div>
</div> 



## 一、直接选择排序
源码
```py
def selection_sort(arr):
    for i in range(len(arr)-1):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

```
**时间复杂度**：选择排序的最优时间复杂度、平均时间复杂度和最坏时间复杂度均为 $O(n^2)$。
**空间复杂度**：算法运行过程中不需要申请额外的空间，故空间复杂度为 $O(1)$
**稳定与否**：在交换两个数时会改变数的相对顺序，故该算法是不稳定的

**优点**：选择排序的优点在于其实现简单，且在数据量较小或部分有序的情况下表现较好。此外，选择排序在每一轮选择过程中只进行一次交换操作，减少了数据移动的次数，对于某些场景下的数据交换成本较高的情况，选择排序可能更为高效。
**缺点**：选择排序的主要缺点是时间复杂度较高，尤其是在处理大规模数据时，效率较低。此外，选择排序不适用于链表等数据结构，因为其需要频繁地访问和交换元素。

---

## 二、锦标赛排序
源码
```py
import sys
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
```
**时间复杂度**：在整个算法运行的过程中需要构建 $n$ 次 锦标赛树，每次构建之后需要 $ \log n $ 次操作选出最小值，故最后的时间复杂度为 $O(nlogn)$ 
**空间复杂度**：可以观察到算法运行过程中申请了额外的长度为 $2n$ 的空间，故该算法的空间复杂度为 $O(n)$
**稳定与否**：该算法是稳定的，在算法运行过程中，有相同的值的叶节点的相对位置没有变化

**优点**：锦标赛排序的优点在于其能够高效地处理大量数据，尤其是在需要频繁进行插入和删除操作的场景中表现出色。此外，锦标赛排序通过构建锦标赛树，可以在较短的时间内找到最小或最大元素，提高了排序效率。
**缺点**：锦标赛排序的主要缺点是其空间复杂度较高，需要额外的存储空间来构建锦标赛树。此外，锦标赛排序的实现相对复杂，对于小规模数据排序时，可能不如其他简单排序算法高效。


<div style="display:flex; gap:2em; justify-content:center;">
  <div style="text-align:center;">
    <img src="./tournament-sort1.png" width="600"/>
  </div>
  <div style="display:flex; gap:2em; justify-content:center;">
    <div style="text-align:center;">
      <img src="./tournament-sort2.png" width="600"/>
    </div>
  </div>
</div>

---

## 三、堆排序
源码
```py
def heap_sort(arr):
    n = len(arr)
    # 在给定的区间内构造大顶堆
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

    # 从最后一个节点的父节点开始 sift down 以完成堆化 
    i = ((n - 1) - 1) / 2
    while i >= 0:
        sift_down(arr, i, n - 1)
        i -= 1
    # 先将第一个元素和已经排好的元素前一位做交换，再重新调整（刚调整的元素之前的元素），直到排序完毕
    i = n - 1
    while i > 0:
        arr[0], arr[i] = arr[i], arr[0]
        sift_down(arr, 0, i - 1)
        i -= 1
```
**时间复杂度**：堆排序的最优时间复杂度、平均时间复杂度、最坏时间复杂度均为
$O(n\log n)$。在建立大顶堆的过程中的时间复杂度为  $\sum_{h=0}^{k} \frac{n}{2^{h+1}} \cdot O(k-h) = O\left(n \sum{t=1}^{k} \frac{t}{2^t}\right) = O(n)$, 在排序时，$\sum_{i=n-1}^{1} O(\log i) = O(n \log n) $ 
**空间复杂度**：因为所有操作都是在原来存储乱序数字的数组上进行的，所以空间复杂度为 $O(1)$
**稳定与否**：由于算法中存在交换操作，故其是不稳定的

**优点**：堆排序的优点在于其时间复杂度稳定，为 $O(n \log n)$，适用于大规模数据排序。此外，堆排序是一种原地排序算法，空间复杂度为 $O(1)$，不需要额外的存储空间。堆排序还具有较好的缓存性能，因为它在内存中连续访问数据。
**缺点**：堆排序的主要缺点是其实现相对复杂，代码较长且不易理解。此外，堆排序在处理小规模数据时，效率可能不如其他简单排序算法（如插入排序或冒泡排序）。另外，堆排序不是稳定的排序算法，对于需要保持相同元素相对顺序的场景不适用。

---

## 四、直接插入排序
直观点的解释：
每次从原数组中取出一个数从最后开始一一比较，大于这个数就后移，小于就插入该位置，直到所有数都插入完成

源码
```py
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key: 
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
```
**时间复杂度**：插入排序的最优时间复杂度为 $O(n)$，在数列几乎有序时效率很高。插入排序的最坏时间复杂度和平均时间复杂度都为 $O(n^2)$。
**空间复杂度**：因为所有操作都是在原来存储乱序数字的数组上进行的，所以空间复杂度为 $O(1)$
**稳定与否**：该算法是稳定的，插入的顺序就是数字在乱序数组中原始的下标的顺序

**优点**：插入排序的优点在于其实现简单，代码易于理解和维护。此外，插入排序在处理小规模数据时表现出色，尤其是当数据已经部分有序时，效率较高。插入排序还具有稳定性，能够保持相同元素的相对顺序。
**缺点**：插入排序的主要缺点是其时间复杂度较高，尤其是在处理大规模数据时，效率较低。插入排序在最坏情况下的时间复杂度为 $O(n^2)$，这使得它不适用于大规模数据排序。此外，插入排序在频繁插入和删除操作的场景中表现不佳，因为每次插入都可能需要移动大量元素。

---

## 五、折半插入排序
源码
```py
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

```
与直接插入排序类似，区别在于二分优化了时间复杂度中的大常数，在数据数量较大时有较大优势
**时间复杂度**：折半插入排序的最优时间复杂度为 $O(n \log n)$，在数列几乎有序时效率很高。折半插入排序的最坏时间复杂度和平均时间复杂度都为 $O(n^2)$。
**空间复杂度**：因为所有操作都是在原来存储乱序数字的数组上进行的，所以空间复杂度为 $O(1)$
**稳定与否**：该算法是稳定的，插入的顺序就是数字在乱序数组中原始的下标的顺序

**优点**：折半插入排序的优点在于其通过二分查找优化了插入位置的查找过程，减少了比较次数，提高了排序效率。与直接插入排序相比，折半插入排序在处理大规模数据时表现更好，尤其是在数据已经部分有序的情况下。  
**缺点**：折半插入排序的主要缺点是其时间复杂度仍然为 $O(n^2)$，在处理大规模数据时效率较低。此外，折半插入排序在频繁插入和删除操作的场景中表现不佳，因为每次插入都可能需要移动大量元素。与直接插入排序相比，折半插入排序的实现稍微复杂一些，代码可读性略有降低。


---

## 六、冒泡排序
源码
```py
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
```
**时间复杂度**：分析算法的运行过程可知，对于长度为n的乱序的数组，需要进行 $n * (1 + n) / 2$ 次操作，故时间复杂度为 $O(n^2)$.
**空间复杂度**：该算法在运行时只需为各个变量创建空间而不要申请额外的空间，故空间复杂度为 $O(1)$
```
7(a) 6 8 7(b) 2 9 5
7(a) 6 8 7(b) 2 5 9
7(a) 6 7(b) 2 5 8 9
6 7(a) 2 5 7(b) 8 9
6 2 5 7(a) 7(b) 8 9
2 5 6 7(a) 7(b) 8 9
```
**稳定与否**：从上面冒泡排序的示例可以看出，冒泡排序是一种稳定的排序算法

**优点**：冒泡排序的优点在于其实现简单，代码易于理解和维护。此外，冒泡排序在处理小规模数据时表现出色，尤其是当数据已经部分有序时，效率较高。冒泡排序还具有稳定性，能够保持相同元素的相对顺序。
**缺点**：冒泡排序的主要缺点是其时间复杂度较高，尤其是在处理大规模数据时，效率较低。冒泡排序在最坏情况下的时间复杂度为 $O(n^2)$，这使得它不适用于大规模数据排序。此外，冒泡排序在频繁插入和删除操作的场景中表现不佳，因为每次插入都可能需要移动大量元素。

---

## 七、快速排序
```py
def quick_sort(arr, start, end):
    if start >= end: return
    mid_value = arr[start]
    low = start, high = end
    while low < high:
        while low < high and arr[high] >= mid_value:
            high -= 1
        arr[low] = arr[high]
        while low < high and arr[low] < mid_value:
            low += 1
        arr[high] = arr[low]
    arr[low] = mid_value
    quick_sort(arr, start, low - 1)
    quick_sort(arr, low + 1, end)
```
**时间复杂度**：快速排序的最优时间复杂度和平均时间复杂度为 $O(n\log n)$，最坏时间复杂度为 $O(n^2)$。
**空间复杂度**：因为递归实现需要申请额外的栈帧，且申请的栈帧总数为 $\lfloor \log _2 n\rfloor $ 故该算法的空间复杂度为 $O(n)$
**稳定与否**：因算法中存在大量交换顺序的操作，故该算法是不稳定的 

**优点**：快速排序的优点在于其平均时间复杂度为 $O(n \log n)$，适用于大规模数据排序。快速排序通过分治策略，将大问题分解为小问题，提高了排序效率。此外，快速排序在内存中连续访问数据，具有较好的缓存性能。
**缺点**：快速排序的主要缺点是其最坏时间复杂度为 $O(n^2)$，在某些情况下（如已经有序的数据）可能表现不佳。此外，快速排序不是稳定的排序算法，对于需要保持相同元素相对顺序的场景不适用。递归实现可能导致栈溢出，尤其是在处理大规模数据时。 

---

## 八、归并排序
源码
```py
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
```
**时间复杂度**：归并排序基于分治思想将数组分段排序后合并，时间复杂度在最优、最坏与平均情况下均为 $\Theta (n \log n)$。
**空间复杂度**：递归时生成的辅助数组与原数组等长且不需要重复生成，故该算法的空间复杂度为 $\Theta (n)$ 
**稳定与否**：算法运行过程中，每一个元素所属的区间的顺序不会改变，故该算法是稳定的

**优点**：归并排序的优点在于其时间复杂度稳定，为 $O(n \log n)$，适用于大规模数据排序。归并排序是一种稳定的排序算法，能够保持相同元素的相对顺序。此外，归并排序在处理链表等数据结构时表现出色，因为它不依赖于随机访问。
**缺点**：归并排序的主要缺点是其空间复杂度较高，需要额外的存储空间来存储辅助数组，空间复杂度为 $O(n)$。此外，归并排序的实现相对复杂，代码较长且不易理解。对于小规模数据排序时，归并排序可能不如其他简单排序算法高效。

---

## 九、希尔排序

排序对不相邻的记录进行比较和移动：
- 将待排序序列分为若干子序列（每个子序列的元素在原始数组中间距相同）；
- 对这些子序列进行插入排序；
- 减小每个子序列中元素之间的间距，重复上述过程直至间距减少为 1

源码
```py
def shell_sort(arr):
    n = len(arr)
    gap = n // 2 # 初始增量设为数组长度的一半 取整
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2
```

```py
def shell_sort_2(array):
    n = len(array)
    h = 1
    while h < n / 3:
        h = int(3 * h + 1)
    while h >= 1:
        for i in range(h, n):
            j = i
            while j >= h and array[j] < array[j - h]:
                array[j], array[j - h] = array[j - h], array[j]
                j -= h
        h = int(h / 3)

```

**算法步骤说明**：
eg.运行实例 按第二种方式生成 第一种没区别的
```cpp
arr = [49, 38, 65, 97, 76, 13, 27, 49, 55, 4, 62, 8, 2, 1, 3, 0]

//step1
h = 13
arr = [49, 38, 65, 97, 76, 13, 27, 49, 55, 4, 62, 8, 2,       
        1,  3,  0]
//交换后
arr = [ 1,  3,  0, 97, 76, 13, 27, 49, 55, 4, 62, 8, 2,
       49, 38, 65]

//step2
h = 4
arr = [  1,  3,  0,  97,
        76, 13, 27, 49,
        55,  4, 62,  8,
         2, 49, 38, 65]
//每一列内进行直接插入排序
arr = [  1,  3,  0,  8,
         2,  4, 27, 49,
        55, 13, 38, 65,
        76, 49, 62, 97]

//step3
h = 1
//gap坍缩到1，就是普通的插入排序
arr = [0, 1, 2, 3, 4, 8, 13, 27, 38, 49, 49, 55, 62, 65, 76, 97]
```

**时间复杂度**：希尔排序的时间复杂度依赖于增量序列的选择。使用简单的增量序列时，最坏时间复杂度为 $O(n^2)$，而使用更复杂的增量序列（如 Knuth 序列）时，时间复杂度可以降低到 $O(n^{3/2})$ 或更好。
以下是将图片中的两个命题内容，转为 Markdown 格式的文本：

若间距序列为 $H = \{2^k - 1 \mid k = 1,2,\dots,\lfloor\log_2 n\rfloor\} \quad \text{（从大到小）}$ 则希尔排序算法的时间复杂度为 $O(n^{3/2})$

若间距序列为 $H = \{k = 2^p \cdot 3^q \mid p,q \in \mathbb{N},\ k \leq n\} \quad \text{（从大到小）}$ 则希尔排序算法的时间复杂度为 $O(n\log^2 n)$

**空间复杂度**：希尔排序在排序过程中只需要常数级别的额外空间，故空间复杂度为 $O(1)$
**稳定与否**：由于希尔排序在排序过程中会交换不相邻的元素，故该算法是不稳定的

**优点**：希尔排序的优点在于其时间复杂度较低，尤其是在处理大规模数据时表现出色。希尔排序通过分组和间隔排序，减少了数据移动的次数，提高了排序效率。此外，希尔排序是一种原地排序算法，空间复杂度为 $O(1)$，不需要额外的存储空间。
**缺点**：希尔排序的主要缺点是其实现相对复杂，代码较长且不易理解。此外，希尔排序不是稳定的排序算法，对于需要保持相同元素相对顺序的场景不适用。希尔排序的性能依赖于增量序列的选择，不同的增量序列可能导致不同的排序效率。

---

## 十、桶排序
!!! : 桶排序的内部排序使用了插入排序，因为其在小规模数据排序时效率较高
!!! : 使用别的排序算法诸如快速排序也是可以的但要考虑时间复杂度和空间复杂度的平衡

源码
```py 
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key: 
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def bucket_sort(arr, bucket_size=5):
    if len(arr) == 0:
        return arr

    min_value = min(arr)
    max_value = max(arr)

    bucket_count = (max_value - min_value) // bucket_size + 1 # 计算桶的数量
    #桶的数量是根据数据的范围和每个桶的大小来确定的。具体来说，桶的数量等于数据范围除以每个桶的大小再加一。

    buckets = [[] for _ in range(bucket_count)]

    for num in arr:
        index = (num - min_value) // bucket_size
        buckets[index].append(num)

    sorted_arr = []
    for bucket in buckets:
        insertion_sort(bucket)  # 使用插入排序对每个桶进行排序
        sorted_arr.extend(bucket)

    arr[:] = sorted_arr
```

**当然这样也可以 这个好懂一点**
```py
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
        #这一步中，存放到哪一个桶里面是根据元素的值决定的，值越大，存放的桶的下标越大
        #所以后面合并的时候也是从小到大合并的

    p = 0
    for i in range(0, n):
        insertion_sort(bucket[i])
        for j in range(0, len(bucket[i])):
            a[p] = bucket[i][j]
            p += 1
```


**时间复杂度**：桶排序的时间复杂度主要取决于桶的数量和每个桶内排序算法的效率。平均情况下，桶排序的时间复杂度为 $O(n + n^2 / k + k)$，其中 $n$ 是元素数量，$k$ 是桶的数量。在最坏情况下，如果所有元素都落入同一个桶中，时间复杂度将退化为 $O(n^2)$。(将值域平均分成 𝑛 块 + 排序 + 重新合并元素)
**空间复杂度**：桶排序需要额外的空间来存储桶，空间复杂度为 $O(n + k)$，其中 $n$ 是元素数量，$k$ 是桶的数量。
**稳定与否**：桶排序是稳定的排序算法，因为在将元素放入桶时，保持了相同元素的相对顺序。

**优点**：桶排序的优点在于其能够高效地处理大量数据，尤其是在数据分布均匀的情况下表现出色。桶排序通过将数据分布到不同的桶中，减少了每个桶内的排序工作量，提高了整体排序效率。此外，桶排序是一种稳定的排序算法，能够保持相同元素的相对顺序。
**缺点**：桶排序的主要缺点是其空间复杂度较高，需要额外的存储空间来存储桶，尤其是在桶的数量较多时。此外，桶排序的性能依赖于数据的分布情况，如果数据分布不均匀，可能导致某些桶内元素过多，影响排序效率。选择合适的桶大小和数量也是一个挑战，需要根据具体数据情况进行调整。

---

## 十一、计数排序
**原理**
计数排序的工作原理是使用一个额外的数组 𝐶，其中第 𝑖 个元素是待排序数组 𝐴 中值等于 𝑖 的元素的个数，然后根据数组 𝐶 来将 𝐴 中的元素排到正确的位置。

它的工作过程分为三个步骤：
- 计算每个数出现了几次；
- 求出每个数出现次数的 前缀和；
- 利用出现次数的前缀和，从右至左计算每个数的排名。

可以看出，如果不使用前缀和，计数排序是不稳定的，而使用了前缀和之后，计数排序就变成了稳定的排序算法。

!!! : 计算前缀和的原因
直接将 𝐶 中正数对应的元素依次放入 𝐴 中不能解决元素重复的情形。

我们通过为额外数组 𝐶 中的每一项计算前缀和，结合每一项的数值，就可以为重复元素确定一个唯一排名：额外数组 𝐶 中每一项的数值即是该 key 值下重复元素的个数，<mark>而该项的前缀和即是排在最后一个的重复元素的排名。

如果按照 𝐴 的逆序进行排列，那么显然排序后的数组将保持 𝐴 的原序（相同 key 值情况下），也即得到一种稳定的排序算法

源码
```py
def counting_sort(arr):
    if len(arr) == 0:
        return arr

    min_value = min(arr)
    max_value = max(arr)
    range_of_elements = max_value - min_value + 1

    count = [0] * range_of_elements
    output = [0] * len(arr)

    for num in arr:
        count[num - min_value] += 1

    for i in range(1, len(count)):
        count[i] += count[i - 1]

    for num in reversed(arr):
        output[count[num - min_value] - 1] = num
        count[num - min_value] -= 1

    arr[:] = output

## 另一种写法
def counting_sort(a, n, w): #n为数组长度，w为数组中最大值
    b = [0] * n
    cnt = [0] * (w + 1)
    for i in range(1, n + 1):
        cnt[a[i]] += 1
    for i in range(1, w + 1):
        cnt[i] += cnt[i - 1]
    for i in range(n, 0, -1):
        b[cnt[a[i]] - 1] = a[i]
        cnt[a[i]] -= 1 # 计算排名 处理重复元素
    return b
```
**时间复杂度**：计数排序的时间复杂度为 $O(n + k)$，其中 $n$ 是输入数组的大小，$k$ 是输入数组中元素的范围（最大值与最小值之差加一）。当 $k$ 远小于 $n$ 时，计数排序非常高效。
**空间复杂度**：计数排序的空间复杂度为 $O(k)$，因为它需要一个大小为 $k$ 的计数数组来存储每个元素的出现次数。
**稳定与否**： 计数排序是稳定的排序算法，因为在将元素放入输出数组时，保持了相同元素的相对顺序。

**优点**：计数排序的优点在于其时间复杂度为 $O(n + k)$，适用于大规模数据排序，尤其是当数据范围较小且元素分布均匀时表现出色。计数排序是一种稳定的排序算法，能够保持相同元素的相对顺序。此外，计数排序的实现简单，代码易于理解和维护。
**缺点**：计数排序的主要缺点是其空间复杂度较高，需要额外的存储空间来存储计数数组，尤其是在数据范围较大时。此外，计数排序仅适用于整数或离散数据，对于连续数据或非整数数据不适用。计数排序在处理数据范围非常大的情况下可能效率较低，因为需要为整个范围分配空间。

--- 

## 十二、基数排序
基数排序（英语：Radix sort）是一种非比较型的排序算法，最早用于解决卡片排序的问题。基数排序将待排序的元素拆分为 𝑘
k 个关键字，逐一对各个关键字排序后完成对所有元素的排序。

如果是从第 1 关键字到第 𝑘 关键字顺序进行比较，则该基数排序称为 MSD（Most Significant Digit first）基数排序；

如果是从第 𝑘 关键字到第 1 关键字顺序进行比较，则该基数排序称为 LSD（Least Significant Digit first）基数排序。

**K-关键字元素的比较**  
用 α_i 表示元素 α 的第 i 关键字。若元素共有 k 个关键字，则对两元素 α 与 b 的默认比较规则如下：

1. 先比较第 1 关键字 α₁ 与 b₁  
   - α₁ < b₁  ⇒ α < b  
   - α₁ > b₁  ⇒ α > b  
   - α₁ = b₁  ⇒ 进入下一步

2. 再比较第 2 关键字 α₂ 与 b₂  
   - α₂ < b₂  ⇒ α < b  
   - α₂ > b₂  ⇒ α > b  
   - α₂ = b₂  ⇒ 进入下一步

⋯

k. 直至第 k 关键字 α_k 与 b_k  
   - α_k < b_k  ⇒ α < b  
   - α_k > b_k  ⇒ α > b  
   - α_k = b_k  ⇒ α = b

**示例**
- 自然数：<mark>个位对齐并在高位补 0 后，从左往右第 i 位即为第 i 关键字。  
- 字符串：按字典序比较时，从左往右第 i 个字符即为第 i 关键字。  
- C++ 的 `std::pair` 与 `std::tuple` 默认比较即采用上述顺序关键字规则。


源码
#LSD
```py   
def counting_sort_for_radix_lsd(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    for i in range(n):
        index = (arr[i] // exp) % 10
        count[index] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    for i in range(n - 1, -1, -1):
        index = (arr[i] // exp) % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1

    for i in range(n):
        arr[i] = output[i]

def radix_sort_lsd(arr):
    max1 = max(arr)
    exp = 1
    while max1 // exp > 0:
        counting_sort_for_radix_lsd(arr, exp)
        exp *= 10
```



**时间复杂度**：基数排序的时间复杂度为 $O(d \cdot (n + k))$，其中 $n$ 是待排序元素的数量，$d$ 是关键字的位数，$k$ 是每个关键字的取值范围。当 $d$ 和 $k$ 都较小且固定时，基数排序可以达到线性时间复杂度 $O(n)$。
**空间复杂度**：基数排序的空间复杂度为 $O(n + k)$，其中 $n$ 是待排序元素的数量，$k$ 是每个关键字的取值范围。主要空间开销来自于用于计数排序的辅助数组。
**稳定与否**：基数排序是稳定的排序算法，因为它在对每个关键字进行排序时，使用了稳定的排序算法（如计数排序），从而保持了相同关键字元素的相对顺序。

**优点**：基数排序的优点在于其时间复杂度可以达到线性时间 $O(n)$，适用于大规模数据排序，尤其是当关键字位数较少且取值范围有限时表现出色。基数排序是一种稳定的排序算法，能够保持相同关键字元素的相对顺序。此外，基数排序可以处理多关键字排序问题，如字符串排序和整数排序。
**缺点**：基数排序的主要缺点是其空间复杂度较高，需要额外的存储空间来存储辅助数组，尤其是在关键字取值范围较大时。此外，基数排序仅适用于整数或字符串等离散数据，对于连续数据或非整数数据不适用。基数排序的实现相对复杂，代码较长且不易理解。








