#ifndef MyBinTree_HPP
#define MyBinTree_HPP

#include "MY_STACK_BASE_ON_SEQLIST.hpp"
#include "MyQueue_SeqList.hpp"

template <typename T>
class BinTreeNode {
    T _data;
    BinTreeNode<T> *_leftChild, *_rightChild;

    template <typename U>
    friend class BinaryTree;

    public:
    BinTreeNode(): _leftChild(nullptr), _rightChild(nullptr) {}
    BinTreeNode(T x, BinTreeNode<T> *l = nullptr, BinTreeNode<T> *r = nullptr): _data(x), _leftChild(l), _rightChild(r) {}
};

template<typename T>
class BinaryTree {
    public:
        BinaryTree() : _root(NULL) {} //构造函数
        BinaryTree(T value) : _RefValue(value), _root(NULL) {} //构造函数
        BinaryTree(BinTreeNode<T>* rootPtr, T refV = T()): _root(rootPtr), _RefValue(refV) {}
        BinaryTree(const BinaryTree<T>& other): _root(nullptr),  _RefValue(other._RefValue)  {if (other._root != nullptr)  _root = Copy(other._root);} //复制构造函数
        ~BinaryTree() { destroy(_root); } //析构函数


        bool IsEmpty() { return _root == NULL; } //判断二叉树是否为空

        BinTreeNode<T> * Parent(BinTreeNode<T> * current){ //返回父结点
            if (_root == NULL || _root == current) return NULL;
            else return Parent(_root, current);
        }

        int Height() { return Height(_root); } //返回树的高度

        //非递归方式返回树的高度
        int Height_nonRecursive() const {
            if (_root == nullptr) return 0;

            MyQueue<BinTreeNode<T>*> Q;
            Q.EnQueue(_root);
            int depth = 0;

            while (!Q.IsEmpty()) {
                int sz = Q.getSize();      // 当前层节点个数
                ++depth;                // 新的一层
                while (sz--) {          // 把当前层全部弹出，并把它们的子节点入队
                    BinTreeNode<T>* cur = Q.DeQueue();
                    if (cur->_leftChild)  Q.EnQueue(cur->_leftChild);
                    if (cur->_rightChild) Q.EnQueue(cur->_rightChild);
                }
            }
            return depth;
        }

        int Size() { return Size(_root); } //返回树的结点数

        BinTreeNode<T> * getRoot() { return _root; } //取根

        void setRoot(BinTreeNode<T> * p) { _root = p; } //修改根
        
        BinTreeNode<T> * Search(T item) {return Search(_root, item);} //搜索
            
        int Insert(T item) {return Insert(_root, item);} //插入新元素

        void createBinTree(T in[]) {createBinTree(in, _root); }//从输入流读入建树

        void printBinTree() {printBinTree(_root);} //用广义表输出二叉树

        void Traverse() {Traverse(_root, 0); cout << endl; cout << endl;} //用凹入表输出二叉树

        void createBinTree_pre(T pref[]) {int n = 0; createBinTree_pre(pref, _root, n);} //用扩展前序序列建树
            
        void PreOrder() {PreOrder(_root);cout << endl;} //前序遍历

        void InOrder() {InOrder(_root);cout << endl;} //中序遍历

        void PostOrder() {PostOrder(_root);cout << endl;} //后序遍历
        
        //层次序遍历 
        //从根节点开始，压入一个结点就把它的子节点(存在的话)压入队列中，同时弹出队首元素进行输出，这样保证了顺序
        void LevelOrder(){
            if (_root == nullptr) return;                 // 空树直接回去
                MyQueue<BinTreeNode<T>*> Q;                   // 1. 存节点指针
                Q.EnQueue(_root);                             // 2. 根入队

                while (!Q.IsEmpty()) {                        // 3. 标准 BFS
                    BinTreeNode<T>* cur = Q.DeQueue();        // 4. 弹出队头
                    if (cur == nullptr) continue;             // 5. 防御：DeQueue空返回

                    cout << cur->_data << ' ' << flush;       // 6. 立即输出+刷新

                    if (cur->_leftChild)  Q.EnQueue(cur->_leftChild);
                    if (cur->_rightChild) Q.EnQueue(cur->_rightChild);
                }
            cout << endl;
        }   
        //不成功的
        // MyQueue<BinTreeNode<T> * > Q;
        // BinTreeNode<T> * p = _root, * temp = nullptr;
        // Q.EnQueue(p);
        // while(!Q.empty()){
        //     temp = Q.DeQueue(); cout << temp->_data << " ";
        //     if(temp->_leftChild) Q.EnQueue(temp->_leftChild);
        //     if(temp->_rightChild) Q.EnQueue(temp->_rightChild);
        // }

        // 非递归前序遍历 
        void PreOrder_iter() {
            if (_root == nullptr) return;
            Stack<BinTreeNode<T>*> S;          
            BinTreeNode<T>* p = _root;  
            S.push(p);                         

            while (!S.empty()) {
                p = S.pop();
                cout << p->_data << ' ';       

                if (p->_rightChild) S.push(p->_rightChild); // 右先压，后出
                if (p->_leftChild)  S.push(p->_leftChild);  // 左后压，先出
            }
            cout << endl;
        }

        //非递归中序遍历
        void InOrder_iter() {
            if (_root == nullptr) return;          
            Stack<BinTreeNode<T>*> S;              
            BinTreeNode<T>* p = _root;             

            while (true) {
                while (p != nullptr) {            
                    S.push(p);
                    p = p->_leftChild;
                }
                if (S.empty()) break;              

                p = S.pop();            // 上一级while遍历完，此时需要回退（父节点）           
                cout << p->_data << ' ' << flush;  

                p = p->_rightChild;                
            }
            cout << endl;
        }

        // 非递归后序遍历
        void PostOrder_iter() {
            if (_root == nullptr) return;

            Stack<BinTreeNode<T>*> S;
            BinTreeNode<T>* p   = _root;
            BinTreeNode<T>* last = nullptr;   // 上一次完整访问的节点

            while (true) {
                while (p != nullptr) {
                    S.push(p);
                    p = p->_leftChild;
                }
                if (S.empty()) break;


                p = S.top();                    
                if (p->_rightChild != nullptr && p->_rightChild != last)
                    p = p->_rightChild;         // 右子树存在且未访问
                else {
                    S.pop();                    // 右子树已访问或为空，可以输出根
                    cout << p->_data << ' ';
                    last = p;                   
                    p = nullptr;                
                }
            }
            cout << endl;
        }

        void VLR_LVR_create(T VLR[], T LVR[], int n) { _root = VLR_LVR_create(VLR, LVR, n, _root); cout << endl;}

        friend bool operator == (const BinaryTree<T>& s, const BinaryTree<T>& t) {return equal(s._root, t._root);} //重载操作符：判等

        friend bool operator != (const BinaryTree<T>& s, const BinaryTree<T>& t) {return !equal(s._root, t._root);}



    protected:
        BinTreeNode<T> * _root; //二叉树的根结点
        T _RefValue; //数据输入停止标志

        void destroy(BinTreeNode<T> * & subTree){//销毁子树
            if(subTree != nullptr){
                destroy(subTree->_leftChild);
                destroy(subTree->_rightChild);
                delete subTree;
                subTree = nullptr;
            }
        }

        BinTreeNode<T>* Copy(BinTreeNode<T>* origNode) { //复制
            if (origNode == nullptr) return nullptr;

            BinTreeNode<T>* newNode = new BinTreeNode<T>(origNode->_data); // 拷贝当前节点（值语义）

            newNode->_leftChild  = Copy(origNode->_leftChild);        // 递归拷贝左右子树
            newNode->_rightChild = Copy(origNode->_rightChild);

            return newNode;
        }
       
        //从广义表单构建二叉树 "A(B(D,E(G,)),C(,F))#"
        void createBinTree(T in[], BinTreeNode<T>*& subTree){ 
            BinTreeNode<T>* nodeStack[30];   BinTreeNode<T>* p = nullptr;   // 当前新节点
            int top = -1; int  k = 0;
            int flag = 0;                 // 1 左孩子  2 右孩子
            T ch = in[k++];         
            subTree  = nullptr;

            while (ch != T('#')) {         // 结束符
                while (ch == ' ') ch = in[k++];
                switch (ch) {
                case '(': nodeStack[++top] = p; flag = 1; break;               // 刚刚创建的 p 入栈，准备读左子树
                case ')': --top; break; // 当前子树读完，弹出栈顶
                case ',': flag = 2; break; // 准备读右子树
                default:                   // 遇到真实数据，new 节点
                    p = new BinTreeNode<T>(ch);
                    if (subTree == nullptr) {          // 第一个节点做根
                        subTree = p;
                    } else {                           // 挂到栈顶父亲下
                        BinTreeNode<T>* par = nodeStack[top];
                        if (flag == 1)  par->_leftChild  = p;
                        else            par->_rightChild = p;
                    }
                }
                ch = in[k++];
            }
        }
        
        //从扩展前序序列构建二叉树 pref = [A, B, C, #, #, D, E, G, #, #, F, #, #, #, ;]
        void createBinTree_pre(T pref[], BinTreeNode<T> * & subTree, int& n){
            T ch = pref[n++];
            if (ch == ';') return;
            if (ch != '#'){
                subTree = new BinTreeNode<T>(ch);
                createBinTree_pre(pref, subTree->_leftChild, n);
                createBinTree_pre(pref, subTree->_rightChild, n);
            }
            else subTree = NULL;
        }

        //利用前序序列和中序序列构造二叉树
        //例：输入：前序：ABHFDECKG；中序：HBDFAEKCG
        BinTreeNode<T>* VLR_LVR_create(T VLR[], T LVR[], int n, BinTreeNode<T>* subtree) {
            if (n == 0) return nullptr;

            int k = 0;
            while (VLR[0] != LVR[k]) k++;

            subtree = new BinTreeNode<T>(VLR[0]);
            subtree->_leftChild = VLR_LVR_create(VLR + 1, LVR, k, subtree->_leftChild);
            subtree->_rightChild = VLR_LVR_create(VLR + k + 1, LVR + k + 1, n - k - 1, subtree->_rightChild);

            return subtree;
        }


        BinTreeNode<T> * Parent(BinTreeNode<T> * subTree, BinTreeNode<T> * current) {  // 求父节点 //subTree 代表搜索的子树的根节点， current代表当前搜索的结点  
            if(subTree == nullptr) return NULL;
            if(subTree == current) return NULL;
            if(subTree->_leftChild == current || subTree->_rightChild == current) return subTree;

            BinTreeNode<T> *p;
            if((p = Parent(subTree->_leftChild, current)) != nullptr) return p;
            else return Parent(subTree->_rightChild,current);
        }

        BinTreeNode<T> * Search(BinTreeNode<T> * p, T item); //搜索
        
        //插入
        int Insert(BinTreeNode<T> * & subTree, T item){
            if (subTree == nullptr) {              // 空位置 → 插入
                subTree = new BinTreeNode<T>(item);
                return 1;
            }
            // 固定策略：优先插左子树，左子树满了再插右子树
            if (Insert(subTree->_leftChild, item)) return 1;
            return Insert(subTree->_rightChild, item);
        }
        
        //求子树的高度
        int Height(BinTreeNode<T> * subTree){
            if(subTree == NULL) return 0;
            else {
                int i = Height(subTree->_leftChild);
                int j = Height(subTree->_rightChild);
                return (i < j)? j + 1: i + 1;
            }
        }
        
        //求子树的结点数
        int Size(BinTreeNode<T> * subTree){
            if(subTree == NULL) return 0;
            else return 1 + Size(subTree->_leftChild) + Size(subTree->_rightChild);
        }
        
        //用广义表输出二叉树
        void printBinTree(BinTreeNode<T> * subTree){
            if(subTree == NULL) cout << "#" ;
            if(subTree != NULL) {
                cout << subTree->_data << "";
                if (subTree->_leftChild || subTree->_rightChild) {
                    cout << "(";
                    printBinTree(subTree->_leftChild);
                    cout << ",";
                    printBinTree(subTree->_rightChild);
                    cout << ")";
                }
            }
        }

        // 凹入表输出
        // 其实就是控制每个结点前面空格的个数 #可不加
        void Traverse(BinTreeNode<T> * subTree, int k) {
            if (subTree == nullptr) { // 空节点也按格式打出，方便对齐
                for (int i = 0; i < k; ++i) cout << "    ";   // 4*k 空格
                cout << "#\n";
                return;
            }

            for (int i = 0; i < k; ++i) cout << "    ";
            cout << subTree->_data << '\n';

            Traverse(subTree->_leftChild,  k + 1);
            Traverse(subTree->_rightChild, k + 1);
        }

        //前序遍历
        void PreOrder(BinTreeNode<T> * subTree){
            if(subTree != NULL) {
                cout << subTree->_data << " ";
                PreOrder(subTree->_leftChild);
                PreOrder(subTree->_rightChild);
            }
        }

        //中序遍历
        void InOrder(BinTreeNode<T> * subTree){ 
            if(subTree != NULL) {
                InOrder(subTree->_leftChild);
                cout << subTree->_data << " ";
                InOrder(subTree->_rightChild);
            }
        }

        //后序遍历
        void PostOrder(BinTreeNode<T> * subTree){
            if(subTree != NULL) {
                PostOrder(subTree->_leftChild);
                PostOrder(subTree->_rightChild);
                cout << subTree->_data << " ";
            }
        } 

        bool equal(BinTreeNode<T> * a, BinTreeNode<T> * b){ //判等
            if (a == nullptr && b == nullptr) return true;
            if (a == nullptr || b == nullptr) return false;

            if (a->_data != b->_data) return false;
            return equal(a->_leftChild, b->_leftChild) && equal(a->_rightChild, b->_rightChild);
        } //传入一个父（根）节点 向下比对

};

#endif
/*
1、定义二叉树(构造函数、析构函数、求根、遍历等)

2、基于用户输入的广义表串、前序扩展序列、中缀算术表达式，建立二叉树

     广义表串：A(B(D,E(G,)),C(,F))

     前序扩展：AB#D##C##

     中缀表达式：a*(b+c)-d

     建议：对输入做合法性判断，并给出相应提示

3、采用凹入表形式输出第2步创建的二叉树内容

4、基于用户输入的前序序列和中序序列，建立二叉树

     例：输入：前序：ABHFDECKG；中序：HBDFAEKCG

5、分别以前序、中序、后序、层次遍历方式，输出第4步创建的二叉树内容

6、把第4步创建的二叉树内容转化为广义表串 */


// 1、编写一个算法求二又树的深度(递归、非递归)。

// 2、设无向图G采用邻接表存储，设计一个算法，采用BFS、DFS求图G的所有连通分量。