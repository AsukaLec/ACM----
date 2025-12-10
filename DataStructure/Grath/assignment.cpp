#include "GraphAL.hpp"

int main() {
    int n = 9;                  
    GraphAL g(n);

    g.addEdge(0, 1);
    g.addEdge(1, 2);
    g.addEdge(0, 2);
    g.addEdge(0, 3);
    // g.addEdge(1, 4);
    g.addEdge(4, 6);
    // g.addEdge(3, 5);
    // g.addEdge(2, 5);
    g.addEdge(5, 7);
    g.addEdge(7, 8);

    cout << "DFS\n";
    g.DFS_Components();

    cout << "\nBFS\n";
    g.BFS_Components();
    return 0;
}

//2、设无向图G采用邻接表存储，设计一个算法，采用BFS、DFS求图G的所有连通分量。

//输出
// DFS
// Component 1 : 0 1 2 3 
// Component 2 : 4 6
// Component 3 : 5 7 8

// BFS
// Component 1 : 0 1 2 3
// Component 2 : 4 6
// Component 3 : 5 7 8