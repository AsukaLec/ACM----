#include <iostream>
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
};

template <typename T>
class SequenceList {
    private:
    T* data;
    int capacity;//容量
    int size;//实际大小

    void expand () { //扩容函数，防止线性表在反复增删后溢出
            if(size < capacity * 0.75) return; // 当装填因子较大时，扩容
            
            capacity *= 2;
            T* oldData = data;
            data = new T[capacity];

            for (int i = 0; i < size; i++) {
                data[i] = oldData[i];
                delete [] oldData;
            }
        }

    void shrink () { //缩容函数，减小线性表的占用空间
        if(size > capacity * 0.25) return; // 当装填因子较小时，缩容
        
        if(capacity >= 2) capacity *= 2;
        T* oldData = data;
        data = new T[capacity];

        for (int i = 0; i < size; i++) {
            data[i] = oldData[i];
            delete [] oldData;
        }
    }

    public :
    SequenceList () : data(new T[10]), capacity(10), size(0) {} //默认构造函数
    SequenceList (SequenceList <T> &other) { //复制构造函数
        size = other->size;
        capacity = other->capacity;
        data = new T[capacity];

        for (int i = 0; i < size; i ++) {
            data[i] = other[i];
        }
    }  
    ~SequenceList() {delete [] data;}; //析构函数

    void insert (T const& n, int position) { //插入
        if (position < 0 || position > size) throw "insert pos error";
        
        ++size;
        expand();
        for (int i = size - 1; i > position; i --) {data[i] = data[i - 1];}
        data[position] = n;
    }

    void remove (int start, int end) { //删除 (start, end) 区间左闭右开 单个元素左右两个端点相同即可
        while (end < size) {
            data[start++] = data[end++ + 1];
        }
        size = start;
        shrink();
    }

    int find(const T& x) const {
        for (int i = 0; i < size; ++i)
            if (data[i] == x) return i;
        return -1;
    }

    void getelm(int pos) {
        if (pos < 0 || pos > size) throw "out of index";
        data[pos].show(); // 展示信息，需确保保存的数据中含有show()方法
    }
    
};

int main() {
    SequenceList <Team> RankList;

    Team ferrari("Ferrari", 1, Driver(16, "Charles Leclerc"), Driver(44, "Lewis Hamilton"));
    ferrari.show();
    Team redbull("Red Bull", 2, Driver(1, "Max Verstappen"), Driver(22, "Yuki Tsunoda"));
    Team mercedes("Mecedes", 3, Driver(63, "Geroge Russal"), Driver(12,"Kimi Antonelli"));

    RankList.insert(ferrari, 0);
    RankList.insert(redbull, 1);
    RankList.insert(mercedes, 2);

    int pos = RankList.find(ferrari);
    cout << "pos = " << pos + 1<< endl;    


    RankList.getelm(pos);
    
    RankList.remove(1, 1);
    RankList.getelm(1);
}