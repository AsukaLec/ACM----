#ifndef MYLINKEDLIST_HPP
#define MYLINKEDLIST_HPP

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

    bool insert (int pos, const T& x) {
        LinkNode<T>* cur = Locate(pos);
        if (cur == NULL) return false;
        LinkNode<T>* newNode = new LinkNode<T> (x);
        newNode->_linknext = cur->_linknext;
        cur->_linknext = newNode;
        return true;
    }

    bool remove(int pos) { 
        LinkNode<T>* cur = Locate(pos);
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
#endif