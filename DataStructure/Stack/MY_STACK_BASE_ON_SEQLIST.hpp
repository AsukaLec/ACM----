#ifndef MYSTACK_HPP
#define MYSTACK_HPP

#include <bits/stdc++.h>

using namespace std;

template <typename T>
class MyVector {
    private: 
    T* _data;
    size_t _size; //size_t == unsigned longlong
    size_t _capacity;

    void check_range(size_t index) {
        if(index >=_size) {
            stringstream ss;
            ss << "out of range, index: " << index << " , size = " << _size;
            throw out_of_range(ss.str());
        }
    }

    public:
    // ====== 迭代器类 ======
    class iterator {
        private:
        T* ptr;

        public:
        iterator() {}
        iterator(T* ptr): ptr(ptr) {}
        
        using value_type = T; //stl会用到
        using iterator_category = std::random_access_iterator_tag; //reverse 会用到 可以随机访问的迭代器
        using difference_type   = std::ptrdiff_t; //计算两个迭代器的距离 
        using pointer           = T*; //什么值的一个pointer
        using reference         = T&; //什么值的类型引用
        // ====== 运算符重载 ======
        iterator &operator ++ ()  { // ++iter;
            ptr ++;
            return *this;//返回自身
        }

        iterator operator ++ (int) { // iter ++；
            iterator temp = *this;
            ptr ++;
            return temp;
        } // 复杂类型，++在前没有产生新数据，性能会好很多

        iterator &operator -- ()  { // ++iter;
            ptr --;
            return *this;//返回自身
        }

        iterator operator -- (int) { // iter ++；
            iterator temp = *this;
            ptr --;
            return temp;
        } // 复杂类型，++在前没有产生新数据，性能会好很多

        T &operator *() const { return *ptr; }
        T *operator ->() const { return ptr; }

        bool operator != (const iterator& other) const { return ptr != other.ptr; }
        bool operator < (const iterator &other) const { return ptr < other.ptr; }
        bool operator > (const iterator &other) const { return ptr > other.ptr; }
        bool operator == (const iterator &other) const { return ptr == other.ptr; }
        iterator operator+ (int n) const { return iterator(ptr + n); }
        iterator operator- (int n) const { return iterator(ptr - n); }

        int operator- (const iterator &other) const { return ptr - other.ptr; }

    };
    iterator begin() { return iterator(_data); }
    iterator end() { return iterator(_data + _size); }
    // ====== ======

    //====== 构造函数们 ======
    MyVector() { // 默认构造函数
        _data = nullptr;
        _size = 0;
        _capacity = 0;
    }

    // MyVector() :_data(nullptr), _size(0) {}

    MyVector(size_t n, T init = T()) { //一个缺省值  T()，用来给MyVector<int>(10, 123)用的 
        _data = new T[n];
        for (size_t i = 0; i < n; i ++) {
            _data[i] = init ;
        }
        _size = n;
        _capacity = n;
    }
    MyVector(const MyVector<T> &other) : _data(new T[other.size()], _size(other.size()), _capacity(other.size())){ //拷贝构造函数
        for (size_t i = 0; i < other.size(); i ++) {
            _data[i] = other[i];
        }
    }
    MyVector(const initializer_list<T> init) :_data(new T[init.size()]), _size(0), _capacity(init.size()){ //列表初始化
        for (T x : init) { //list不能下标访问，只能范围循环
            _data[_size++] = x;
        }
    }
    ~MyVector() { delete[] _data; } 
    // ====== ======

    // ====== 运算符重载 ======
    const T &operator[](int index) const {  return _data[index]; } //索引访问，返回引用以修改 const 对象调用 且返回的对象不可以被修改
    T &operator[](int index) { return _data[index];} //非 const 对象调用

    const T& at(size_t index) const { //at 访问，下标溢出检测
        check_range(index);
        return _data[index];
    }
    T &at(size_t index) { //at 访问，下标溢出检测
        check_range(index);
        return _data[index];
    }
    
    MyVector<T>& operator = (const MyVector<T> &other) {
        if (this != &other) {
            delete[] _data;
            _data = new T[other.size()];
            _size = other.size();
            _capacity = other.capacity();
            for (size_t i = 0; i < _size; i ++) _data[i] = other[i];
        }
        return *this;
    }


    // ====== ======

    // ====== 接口们 ======

    size_t size() const {return _size;}
    size_t capacity() const {return _capacity;}

    void reserve(size_t new_capacity) { //预存空间，同时保存原有的内容
        if (new_capacity > _capacity) {
            _capacity = new_capacity;
            T *new_data = new T[_capacity];
            for (size_t i = 0; i < _size; i ++) new_data[i] = _data[i];
            delete[] _data;
            _data = new_data;
        }
    }

    void push_back (T elem) { //复杂度也是O(1)哦
        if (_size == _capacity) {
            reserve(_capacity == 0 ? 1 : _capacity * 2);
        }
        _data[_size++] = elem;
    }

    void resize(size_t _new_size, T init = T()) { //更改已有的data的大小，缺失值补齐
        if (_new_size > _capacity) reserve(_new_size);
        for (size_t i = _size; i < _new_size; i ++) {
            _data[i] = init;
        }
        _size = _new_size;
    }

    T pop_back() {
        if (_size == 0) throw out_of_range("can not pop_back empty vector");
        T cur = _data[_size - 1];
        _size --;

        return cur;
    }

    void clear() { _size = 0; }

    void insert(const iterator & pos, T x) {
        size_t index = pos - begin();
        if (_size == _capacity) reserve(_capacity == 0 ? 1 : _capacity * 2);
        for (size_t i = _size; i > index; i --) _data[i] = _data[i - 1];
        _data[index] = x;
        _size ++;
    }
 
    void erase(const iterator &pos) {
        for (size_t i = pos - begin(); i < _size - 1; i ++) {
            _data[i] = _data[i + 1];
        }
        _size --;
    }

    void erase(const iterator &start, const iterator &end) { // [star, end)
        size_t diff = end - start; // 注意end > start 
        for (size_t i = start - begin(); i < _size - diff; i ++) {
            _data[i] = _data[i + diff];
        } 
        _size -= diff;
    }

    T &front() { return _data[0]; }
    const T &front() const { return _data[0]; }
    T &back () { return _data[_size - 1]; }
    const T &back() const { return _data[_size - 1]; }

    bool empty() const { return _size == 0; }

    template <typename ... Args>
    void emplace_back(const Args&... args) { //emplace_back 支持多种数据类型的插入
        if (_size == _capacity) reserve(_capacity == 0 ? 1 : _capacity * 2);
        _data[_size++] = T(args...);
    }
    // ====== ======
};

template <class T> 
class Stack : public MyVector<T> {
    public:
    void push (T const& e) { this-> insert(this -> end(), e); }
    T pop() {
        if (this->empty()) throw std::out_of_range("stack empty on pop()");
        return this-> pop_back();}
    T& top() {
        if (this->empty()) throw std::out_of_range("stack empty on top()");
        return (*this) [this->size() - 1];}
    bool empty () { return this->size() == 0; }
    long long stacksize() { return this->size(); } 
};


#endif 