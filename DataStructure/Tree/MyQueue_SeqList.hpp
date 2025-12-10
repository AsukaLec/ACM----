// 循环队列
#ifndef MyQueue_SeqList_hpp
#define MyQueue_SeqList_hpp

// #include "MyVector.hpp"

template <typename T> class MyQueue :public MyVector<T> {
    protected:
        size_t _front, _rear;
        size_t _maxSize;
        MyVector<T> _elements; //用顺序表实现队列

public:
    MyQueue() :_front(0), _rear(0), _maxSize(1e6) {_elements = MyVector<T>(_maxSize);}
    MyQueue(size_t sz, T init = T()) :_front(0), _rear(0), _maxSize(sz) {_elements = MyVector<T>(_maxSize, init);}
    MyQueue(const MyQueue<T> &other) : _elements(other._elements), _front(other._front), _rear(other._rear), _maxSize(other._maxSize) {}
    MyQueue(const initializer_list<T> init) : _front(0), _rear(init.size()), _maxSize(init.size() * 2){_elements = MyVector<T>(init);} //列表初始化后扩容一次，方便增加元素
    ~MyQueue() = default;

    // void test () {
    //     while(!_elements.empty()) {
    //        int i =  _elements.pop_back();
    //        cout << i << " ";
    //     }
    // }

    void makeEmpty() {_front = 0, _rear = 0;}

    bool IsEmpty() {return _front == _rear;}
    bool IsFull() const {return (_rear + 1) % _maxSize == _front;}

    int getSize() const {return (_rear - _front + _maxSize) % _maxSize;}

    bool EnQueue(T x) {
        if(IsFull()) return false;
        _elements[_rear] = x;
        _rear = (_rear + 1) % _maxSize;
        return true;
    }

    T DeQueue() {
        if(IsEmpty()) return nullptr;
        T x = _elements[_front];
        _front = (_front + 1) % _maxSize;
        return x;
    }

    friend ostream& operator << (ostream& os, MyQueue<T>& Q) {
        os << "front = " << Q._front << ", rear = " << Q._rear << endl;
        for (size_t i = Q._front; i != Q._rear; i = (i + 1) % Q._maxSize) os << Q._elements[i] << " ";
        os << endl;
        return os;
    }

};

#endif