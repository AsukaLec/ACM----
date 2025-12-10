# KMP - NEXT
KMP算法的关键在于 `NEXT` 数组的生成
这个过程可以是暴力的，也可以是递推的

##  why NEXT ?
在KMP算法中，我们希望在 T 字符串（即目标字符串）中的遍历顺序是不回退的以降低复杂度，那么就需要根据在先前的比较中得到的信息以优化遍历的方式。
``` cpp
T_str = "ABABABC"
P_str = "ABABC"
```
定义两个指针`T_index` `P_index`分别代表查询到位置。

很容易可以看出，`T_str` `P_str` 的前四位是相同的，而第五位是不同的。
这在 BF 算法中，我们会把  `T_index` 前进`1` 位，  `P_str` 移到索引为 `0` 的位置。

但实际上这是没必要的，因为我们已经知道了在之前的匹配中，`T_str` 中有重复出现的一组字母。
在这个例子中，只用让 `P_str` 前进 `4 - 2 = 2` 位即可，`4`代表目前 `T_index` 的位置， `2` 代表 重复的字母串的长度。

```cpp
"A B A B A B C"
"A B A B C"
```
```cpp
"A B A B A B C"
    "A B A B C"
```

``` cpp
long long find_KMP(const MyString& pat, size_t start = 0) const {
    if (pat._length == 0) return 0;                   
    if (pat._length > this->_length) return -1;       

    size_t i = start;   
    size_t j = 0;       

    while (i < this->_length) {                       
        if (j == pat._length)                         
            return static_cast<long long>(i - j);
        if (j == 0 || this->_str[i] == pat._str[j]) { 
            ++i;
            ++j;
        } else {
            j = pat._next[j - 1];                     
        }
    }
    return (j == pat._length) ? static_cast<long long>(i - j) : -1;
}
```

至于`2` 如何得到？ 这就需要生成一个 `NEXT` 数组了
### generate NEXT

!!! : 简单来说， 假设有一个字符串 `A` 以及它的 `NEXT` 数组， 对于任意一位 `A[x]` ，我们设 `temp = next[x - 1]`，以此来比较 `A[x]` 和 `A[temp]`。`A[temp]` 指向的是目前最长前缀后缀相等的字符串的下一位，二者进行比较，相等的话代表这个相等的子字符串下一位仍有效，故长度加一， `next[x] = next[x - 1]`。不相等的话，就进行<mark>递归（或者叫回退？）</mark>。先思考一下上一位的 `next` 数组是如何得出的，是比较了再上一位的 `next` 数组的值并进行如上过程得出的。这个操作会将 `A[x - 1]` 与 `A[next[x - 2]]`的字符进行比较。现在我们把 `A[x]` 放到 `A[x - 1]` 的位置上比较就好了。具体过程看下面：

先看一下这个例子: 
1. 直接就是重复的情况
``` cpp
"A B A C A B A B"
"0 0 1 0 1 2 ?"
```
此时最后一位和前缀的第三位相等，非常幸运。
我们只需要把前一个next + 1 再赋值给当前位置就好了，` ？` 就等于 3
这代表了相同的前缀和后缀的长度增长了

2. 需要回退，但不会回退到底的情况
``` cpp
"A B A C A B A B"
"0 0 1 0 1 2 3 ?"
```

很不幸，此时最后一位和前缀的第四位不相等，此时我们就需要递归回退了
``` cpp
"A B A   B" // 替代原来 C 的位置
"0 0 1   ?
```
前一位next 的值为1， 代表前面三位最长的前缀和后缀匹配的就是1位，所以说现在要把 `B` 和前面字符串的第二位进行比较（索引为1）

``` cpp
"A B  B"
"0 1  ?"
```
现在要做就是将 `B(A[x])` 和 `B(A[temp])` 比较，结果是相等，那就重复过程1就好了。

3. 要回退到最后一位切仍不相等
``` cpp
"A B  D"
"0 1  ?"
```
重复过程2
``` cpp
"A    D"
"0    ?"
```
显然二者不相等，且`next[temp]`为0，再回退下去就死循环了，那就到这吧，让`next[x] = 0`就好了
``` cpp
    //递推生成Next数组
void getNext() { //next数组的定义为：next[i]表示模式串A[0]至A[i]这个字串，使得前k个字符等于后k个字符的最大值，特别的k不能取i+i
    _next = new long long[_length];   
    _next[0] = 0;                     

    size_t index = 1;   
    size_t temp  = 0;   // 前一个最长前后缀长度

    while (index < _length) {
        if (_str[index] == _str[temp]) {
            _next[index] = temp + 1;
            ++index; ++temp;
        }
        else { // 失配
            if (temp == 0) {
                _next[index] = 0;
                ++index;          // 只推进 index
            } else {
                temp = _next[temp - 1];   // 继续回退（temp>0 保证不越界）
            }
        }
    }
}
```
