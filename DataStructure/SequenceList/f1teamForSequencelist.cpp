#include <iostream>
#include <string>
using namespace std;

//---------------- 车手 ----------------
struct Driver {
    int number;          
    string name;            

    Driver() = default;
    Driver(int no, const string& n) : number(no), name(n) {}
};

//---------------- 车队 ----------------
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

    // 按车队名查找
    bool operator == (const Team& other) const {
        return teamName == other.teamName;
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
};


//---------------- 顺序表（手工数组版） ----------------
template<typename T>
class SeqList {
private:
    T*  data;
    int capacity;
    int size;

    void resize() {
        int newCap = (capacity == 0) ? 4 : capacity * 2;
        T* tmp = new T[newCap];
        for (int i = 0; i < size; ++i)  tmp[i] = data[i];
        delete[] data;
        data = tmp;
        capacity = newCap;
    }

public:
    SeqList() : data(nullptr), capacity(0), size(0) {}
    ~SeqList() { delete[] data; }

    int  getSize()   const { return size; }
    bool isEmpty()   const { return size == 0; }

    T    getElem(int i) const {
        if (i < 0 || i >= size) throw "Index error";
        return data[i];
    }

    void insert(int i, const T& x) {
        if (i < 0 || i > size) throw "Index error";
        if (size == capacity) resize();
        for (int k = size; k > i; --k) data[k] = data[k - 1];
        data[i] = x;
        ++size;
    }

    void remove(int i) {
        if (i < 0 || i >= size) throw "Index error";
        for (int k = i; k < size - 1; ++k) data[k] = data[k + 1];
        --size;
    }

    int find(const T& x) const {
        for (int i = 0; i < size; ++i)
            if (data[i] == x) return i;
        return -1;
    }

    void print() const {
        for (int i = 0; i < size; ++i) {
            const Team& t = data[i];
            std::cout << "#" << t.rank << " " << t.teamName
                      << "  (" << t.drivers[0].number << ")" << t.drivers[0].name
                      << "  (" << t.drivers[1].number << ")" << t.drivers[1].name
                      << "\n";
        }
        std::cout << std::endl;
    }
};

//---------------- 主函数：演示 ----------------
int main() {
    SeqList<Team> list;

    Team ferrari("Ferrari", 1,
                 Driver(16, "Charles Leclerc"),
                 Driver(55, "Carlos Sainz"));

    Team redbull("Red Bull", 2,
                 Driver(1, "Max Verstappen"),
                 Driver(11, "Sergio Perez"));

    Team merc("Mercedes", 3,
              Driver(44, "Lewis Hamilton"),
              Driver(63, "George Russell"));

    list.insert(0, ferrari);
    list.insert(1, redbull);
    list.insert(2, merc);

    std::cout << "---- 当前车队顺序表 ----\n";
    list.print();

    std::cout << "---- 查找 Ferrari ----\n";
    int pos = list.find(ferrari);
    if (pos != -1) cout << "Ferrari 在索引 " << pos << "\n\n";

    std::cout << "---- 删除 Red Bull ----\n";
    list.remove(1);
    list.print();

    return 0;
}
