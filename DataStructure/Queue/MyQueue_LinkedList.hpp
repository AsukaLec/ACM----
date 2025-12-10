
#ifndef MyQueue_LinkedList_hpp
#define MyQueue_LinkedLIst_hpp

#include "MyLinkedList.hpp"


template <typename T> class MyQueue :public List<T> {
    protected:
        LinkNode<T>  *_rear;
        size_t _size;

public:
    MyQueue(): _rear(nullptr), _size(0) {new List<T>;}
    MyQueue(const MyQueue<T> &other) : _size(other->_size), List<T>(other) {
        if (this->_first->_linknext == nullptr) { _rear = nullptr; return; }
        LinkNode<T> *p = this->_first;
        while (p->_linknext != nullptr) p = p->_linknext;
        _rear = p;
    }
    ~MyQueue() = default;



    void makeEmpty1() {
        this->makeEmpty();
        _size = 0;
    }

    bool IsEmpty() {return _size == 0;}
    size_t getSize() const {return _size;}

    void EnQueue(const T &x) {
        LinkNode<T> *node = new LinkNode<T>(x);
        if (_size == 0) {               
            this->_first->_linknext = node;
            _rear = node;
        } else {                        
            _rear->_linknext = node;
            _rear = node;
        }
        ++_size;
    }

    T DeQueue() {
        if (IsEmpty()) throw std::runtime_error("Queue underflow");
        LinkNode<T> *del = this->_first->_linknext;
        T val = del->_data;
        this->_first->_linknext = del->_linknext;
        if (--_size == 0) _rear = nullptr;   
        delete del;
        return val;
    }

    friend std::ostream& operator<<(std::ostream &os, const MyQueue<T> &q) {
        LinkNode<T> *p = q._first->_linknext;
        while (p) {
            os << p->_data << (p->_linknext ? " " : "");
            p = p->_linknext;
        }
        return os;
    }
};

#endif