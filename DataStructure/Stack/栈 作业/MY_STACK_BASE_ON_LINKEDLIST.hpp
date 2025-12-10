#ifndef MY_STACK_BASE_ON_LINKEDLIST_HPP
#define MY_STACK_BASE_ON_LINKEDLIST_HPP

#include <bits/stdc++.h>
using namespace std;

template <class T>
struct LinkNode {
    T _data;
    LinkNode * _linknext;
    LinkNode(LinkNode<T> * ptr = NULL) {_linknext = ptr;}
    LinkNode(const T& item, LinkNode<T> *ptr = NULL) {_data = item, _linknext = ptr;}
};

template <class T> 
class List{
    protected:
    LinkNode<T> *_first;

    public:
    List(){_first = new LinkNode<T>;}//默认构造
    List(const T &x) {_first = new LinkNode<T> (x);}//有参构造
    List(List <T> &l) { // 拷贝构造
        T data;
        LinkNode<T> * srcptr = l._first;
        LinkNode<T> * destptr = _first = new LinkNode<T> ();

        while (srcptr->_linknext != NULL) {
            data = srcptr->_linknext->_data;
            destptr->_linknext = new LinkNode <T> (data);
            destptr = destptr->_linknext;
            srcptr = srcptr->_linknext;
        }

        destptr->_linknext = NULL;
    }
    
    void makeEmpty() {
        LinkNode <T>* q;
        while(_first->_linknext != NULL) {
            q = _first->_linknext;  
            _first = q->_linknext;
            delete q;
        }
    }
    ~List(){makeEmpty();}
    
    int Size() {
        LinkNode <T>* ptr = _first->_linknext;
        int cnt = 0;
        while (ptr != NULL) {
            ptr = ptr->_linknext;
            cnt ++;
        }
        return cnt;
    }

    LinkNode<T>* getHead() {return _first;}

    LinkNode<T>* Search(T x) { //确定某个元素是否在表中
        LinkNode<T>* cur = _first->_linknext;
        while (cur != NULL) {
            if (cur->_data == x) break; //需在结构体 T 中重载 == 运算符
            else cur = cur->_linknext;
        }
        return cur;
    }

    LinkNode<T>* Locate(int i) { //定位某个元素
        if (i < 0) throw "out of index";
        LinkNode<T>* cur = _first; int k = 0;
        while (cur != NULL && k < i) {
            cur = cur->_linknext; k ++;
        }
        return cur;
    }

    bool insert (int i, const T& x) {
        LinkNode<T>* cur = Locate(i);
        if (cur == NULL) return false;
        LinkNode<T>* newNode = new LinkNode<T> (x);
        newNode->_linknext = cur->_linknext;
        cur->_linknext = newNode;
        return true;
    }

    bool remove(int i) {
        LinkNode<T>* cur = Locate(i);
        LinkNode<T>* del = cur->_linknext;
        cur->_linknext = del->_linknext;
        delete del;
        return true;
    }

    bool change(const T& x1, const T& x2) { //查找元素x1 并替换成x2
        LinkNode<T>* cur = Search(x1);
        cur->_data = x2;
        return true;
    } 

    void show() {
        LinkNode<T>* cur = _first->_linknext;
        while(cur != NULL) {
            cout << cur->_data << " "; // 重载 << 运算符
            cur = cur->_linknext;
        }
        cout << endl;
    }

};

template <class T>
class Stack : public List<T> {
private:
    LinkNode<T>* _tail = nullptr; // 维护尾指针
public:
    Stack() : List<T>() {
        _tail = this->_first;
    }
    ~Stack() {
        while (!empty()) {
            pop();
        }
    }

    void push(T const& e) {
        LinkNode<T>* newNode = new LinkNode<T>(e); //默认 _linknext == nullptr
        if (this->Size() == 0) {
            this->_first->_linknext = newNode;
            _tail = newNode;
        } else {
            _tail->_linknext = newNode;
            _tail = newNode; //更新尾指针，让 _tail->linknext 指向空结点
        }
    }

    T pop() {
        if (this->Size() == 0) throw std::runtime_error("Stack underflow");
        LinkNode<T>* prev = this->_first;
        // 只有一个元素
        if (prev->_linknext == _tail) {
            T val = _tail->_data;
            delete _tail;
            prev->_linknext = nullptr;
            _tail = prev;
            return val;
        }
        // 多于一个元素
        while (prev->_linknext != _tail) {
            prev = prev->_linknext;
        }
        T val = _tail->_data;
        delete _tail;
        prev->_linknext = nullptr;
        _tail = prev;
        return val;
    }

    T& top() {
        if (this->Size() == 0) throw std::runtime_error("Stack is empty");
        return _tail->_data;
    }

    bool empty() {
        return this->Size() == 0;
    }

    long long stacksize() {
        return this->Size();
    }
};

#endif