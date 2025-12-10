#include <bits/stdc++.h>
using namespace std;


struct Driver {
    int number;          
    string name;            

    Driver() = default;
    Driver(int num, const string& n) : number(num), name(n) {}
};

struct Team {
    std::string teamName;        
    int rank{};          
    Driver drivers[2];      

    Team() = default;
    Team(const std::string& tn,
         int r,
         const Driver& d1,
         const Driver& d2)
        : teamName(tn), rank(r) {
        drivers[0] = d1;
        drivers[1] = d2;
    }

    Team(const Team& other)
        : teamName(other.teamName),
          rank(other.rank) {
        drivers[0] = other.drivers[0];
        drivers[1] = other.drivers[1];
    }

    bool operator == (const Team& other) const {
        return teamName == other.teamName; //可修改，根据车队排名进行对比等，不唯一
    }
    
    Team operator = (const Team& other) { 
        if (this != &other) {               
            teamName = other.teamName;      
            rank = other.rank;
            drivers[0] = other.drivers[0];  
            drivers[1] = other.drivers[1];
        }
        return *this;                       
    }

    void show () {
        cout << "Team Name: " << teamName << ", Rank : " << rank << ", Driver1: " << drivers[0].name << ", Number: " << drivers[0].number << ", Driver2: " << drivers[1].name << ", Number: " << drivers[1].number << endl;
    }

    friend  ostream & operator<< (ostream& out, const Team& B) {
        out<< "Team Name: " 
            << B.teamName 
            << ", Rank : " 
            << B.rank 
            << ", Driver1: " 
            << B.drivers[0].name 
            << ", Number: " 
            << B.drivers[0].number 
            << ", Driver2: " 
            << B.drivers[1].name 
            << ", Number: " 
            << B.drivers[1].number 
            << endl;
        return out;
    }

};


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

int main() {
    List<int> MyList;
    for(int i = 0; i < 6; i ++) {
        MyList.insert(i, (i+1) * 10);
    }
    MyList.show();

    MyList.change(20, 40);
    MyList.show();

    MyList.remove(3);
    MyList.show();

    MyList.insert(3, 40);
    MyList.show();


    List <Team> RankList;

    Team ferrari("Ferrari", 1, Driver(16, "Charles Leclerc"), Driver(44, "Lewis Hamilton"));
    ferrari.show();
    cout << endl;
    
    Team redbull("Red Bull", 2, Driver(1, "Max Verstappen"), Driver(22, "Yuki Tsunoda"));
    Team mercedes("Mecedes", 3, Driver(63, "Geroge Russal"), Driver(12,"Kimi Antonelli"));

    RankList.insert(0, ferrari);
    RankList.insert(1, redbull);
    RankList.insert(2, mercedes);
    RankList.show();

    RankList.remove(2);
    RankList.show();

    Team mclaren("McLaren", 3, Driver(81, "Oscar Piastri"), Driver(4, "Lando Norris"));
    RankList.insert(2, mclaren);
    RankList.show();

    RankList.change(mercedes, mclaren);
    RankList.show();
}