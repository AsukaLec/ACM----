#include "MyQueue_LinkedList.hpp"

int main() {
    int n; 
    cin >> n;
    MyQueue<int> q;
    cout << 1 << endl << "1 1";
    q.EnQueue(0);q.EnQueue(1);q.EnQueue(1);
    int s = q.DeQueue();
    int t = q.DeQueue();
    for (int i = 1; i <= n; i ++) {
        cout << endl;
        q.EnQueue(0);
        for (int j = 1; j <= i + 2; j ++) {
            q.EnQueue(s + t);
            cout << s + t << " ";
            s = t;
            t = q.DeQueue();
        }                                                                                                                                                                                                                                                                                                                                                                                   
    }
    cout << endl << endl;
    cout << q;
}