#ifndef MYHUFFMANTREE_HPP
#define MYHUFFMANTREE_HPP

#include <iostream>
#include <vector>
#include <map>
#include <stack>
#include <unordered_map>
#include <algorithm>
#include <string>
#include <istream>
#include <fstream>

using namespace std;

#define leafNumber 20
#define totalNumber 100
#define maxValue 999

template<class T, class E>
struct HFNode {
    E _data;
    T _key;
    int _leftchild, _rightchild, _parent;
};

template<class T, class E>
struct HFTree {
    HFNode<T,E> _elem[totalNumber];
    int _num;
    int _root;
    unordered_map<E, string> codeMap;

    // char ch[] = {'A','B','C','D','E','F','G','H'};
    // int  fr[] = { 5 , 9 , 12, 13, 16, 45, 1 , 2 };
    void createHFTree(E ch[], T fr[], int n) {
        // 1. 初始化所有叶子
        for (int i = 0; i < n; ++i) {
            _elem[i]._data  = ch[i];
            _elem[i]._key   = fr[i];
            _elem[i]._parent = _elem[i]._leftchild = _elem[i]._rightchild = -1; //初始化全部置空
        }
        // 2. 不断合并两棵最小树
        for (int i = n; i < 2 * n - 1; ++i) { //生成内部节点
            int s1 = -1, s2 = -1;
            T min1 = maxValue, min2 = maxValue; //第一大和第二大
            for (int k = 0; k < i; ++k) {
                if (_elem[k]._parent == -1) {          // 只在森林里的根里选
                    if (_elem[k]._key < min1) {
                        min2 = min1; s2 = s1; //更新第二大
                        min1 = _elem[k]._key; s1 = k;
                    } else if (_elem[k]._key < min2) {
                        min2 = _elem[k]._key; s2 = k;
                    }
                }
            }
            // 新建内部结点 i
            _elem[s1]._parent = i;
            _elem[s2]._parent = i;
            _elem[i]._leftchild  = s1;
            _elem[i]._rightchild = s2;
            _elem[i]._key        = min1 + min2;
            _elem[i]._parent     = -1;   
        }
        _num  = n;
        _root = 2 * n - 2;   // 最后一个结点即整棵树的根
    }


    void printHFTree() {
        for (int i = 0; i < 2 * _num - 1; ++i) {
            E value = (i < _num) ? _elem[i]._data : '-';
            cout << (i < 10 ? " " : "") << i << ": " << value
                 << (_elem[i]._key < 10 ? " " : "") << " " << _elem[i]._key
                 << "  L=" << _elem[i]._leftchild
                 << "  R=" << _elem[i]._rightchild
                 << "  P=" << _elem[i]._parent << '\n';
        }
        cout << "root = " << _root << "\n\n";
    }

    void printIndexAndKeys() {
        for (int i = 0; i < 2 * _num - 1; ++i)
            cout << i << (i + 1 == 2 * _num - 1 ? '\n' : ' ');

        for (int i = 0; i < 2 * _num - 1; ++i)
            cout << _elem[i]._key << (i + 1 == 2 * _num - 1 ? '\n' : ' ');
        
        cout << endl;
    }

    bool isLeaf(HFNode<T,E> x) {return x._leftchild == -1 && x._rightchild == -1;}

    void generateHuffmanCode() {
        generateHuffmanCode(_root, "");
    }

    void generateHuffmanCode(int root, string code) {
        if(isLeaf(_elem[root])) {
            codeMap.insert({_elem[root]._data, code});
            return;
        }
        generateHuffmanCode(_elem[root]._leftchild, code + "0");
        generateHuffmanCode(_elem[root]._rightchild, code + "1");
    }



    void showHuffmanCode() const {
        for (auto& p : codeMap)
            cout << p.first << ": " << p.second << '\n';
    }


};

#endif 
