#include <iostream>
#include <string>
using namespace std;

/*----------------  Driver  ----------------*/
struct Driver {
    int  number;
    string name;
    Driver() = default;
    Driver(int n, const string& s) : number(n), name(s) {}
};

/*----------------  Team  ----------------*/
struct Team {
    string teamName;
    int    rank{};
    Driver drivers[2];

    Team() = default;
    Team(const string& tn, int r, const Driver& d1, const Driver& d2)
        : teamName(tn), rank(r) { drivers[0] = d1; drivers[1] = d2; }

    /* 只要车队名一样就认为是同一车队，可改成更复杂的规则 */
    bool operator==(const Team& rhs) const {
        return teamName == rhs.teamName;
    }

    void show() const {
        cout << "Team: " << teamName
             << "  Rank: " << rank
             << "  Driver1: " << drivers[0].name << "(" << drivers[0].number << ")"
             << "  Driver2: " << drivers[1].name << "(" << drivers[1].number << ")\n";
    }
};

/*----------------  SequenceList  ----------------*/
template<typename T>
class SequenceList {
private:
    T*  data;
    int capacity;
    int size;

    void expand() {
        if (size < int(capacity * 0.75)) return;
        capacity *= 2;
        T* old = data;
        data = new T[capacity];
        for (int i = 0; i < size; ++i) data[i] = old[i];
        delete[] old;
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

public:
    SequenceList() : data(new T[10]), capacity(10), size(0) {}
    ~SequenceList() { delete[] data; }

    /* 插入：pos 为 0..size */
    void insert(const T& x, int pos) {
        if (pos < 0 || pos > size) throw "insert pos error";
        expand();
        for (int i = size; i > pos; --i) data[i] = data[i - 1];
        data[pos] = x;
        ++size;
    }
    void remove (int start, int end) { //删除 (start, end) 区间左闭右开 单个元素左右两个端点相同即可
        while (end < size) {
            data[start++] = data[end++ + 1];
        }
        size = start;
        shrink();
    }


    /* 按值查找，返回下标或 -1 */
    int find(const T& x) const {
        for (int i = 0; i < size; ++i)
            if (data[i] == x) return i;
        return -1;
    }

    /* 拿到元素并打印（调 show） */
    void getelm(int pos) const {
        if (pos < 0 || pos >= size) throw "getelm pos error";
        data[pos].show();
    }
};

/*----------------  main  ----------------*/
int main() {
    SequenceList<Team> rankList;

    /* 1. 构造三份车队对象 */
    Team ferrari ("Ferrari",  1, Driver(16, "Charles Leclerc"),  Driver(44, "Lewis Hamilton"));
    Team redBull ("Red Bull", 2, Driver(1,  "Max Verstappen"),   Driver(22, "Yuki Tsunoda"));
    Team mercedes("Mercedes", 3, Driver(63, "George Russell"), Driver(12, "Kimi Antonelli"));

    /* 2. 插到表里 */
    rankList.insert(ferrari,  0);
    rankList.insert(redBull,  1);
    rankList.insert(mercedes, 2);

    int pos = rankList.find(redBull);   
    cout << "pos = " << pos << endl;    

    rankList.getelm(pos);
    
    rankList.remove(1, 1);
    rankList.getelm(1);

    Team Williams(mercedes);
}
