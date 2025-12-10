// 1、基于用户输入的字符串，建立霍夫曼编码树，并输出编码方案

// 例：输入：CASTCASTSATATATASA；输出：A：0；T：10；C：110；S：111

// 2、基于编码方案，把输入的字符串转换为编码输出到屏幕，并保存到txt文件

// 3、从txt文件读取编码，并根据编码方案进行解码，输出解码后的字符串

#include "MyHuffmanTree.hpp"
#include <vector>
using namespace std;

int main() {
    cout << "input string: ";
    string target;
    cin >> target;

    map<char, int> freq;
    for (char ch : target) ++freq[ch];

    vector<char> ch;
    vector<int>  fr;
    for (auto& p : freq) {
        ch.push_back(p.first);
        fr.push_back(p.second);
    }
    int n = ch.size();         

    HFTree<int, char> tree;
    tree.createHFTree(ch.data(), fr.data(), n);

    // /* 3. 输出 */
    // cout << "---------- Tree ----------\n";
    // tree.printHFTree();

    // cout << "---------- index and value ----------\n";
    // tree.printIndexAndKeys();

    tree.generateHuffmanCode();
    tree.showHuffmanCode();   

    for(char x : target) {cout << tree.codeMap[x];}

    ofstream fout("huffman_code.txt", ios::out);   
    for (char x : target) fout << tree.codeMap.at(x);
    fout << '\n';
    for (auto& p : tree.codeMap)
        fout << p.first << ": " << p.second << '\n';
    fout.close();
    
    //反编译 构建一个反过来的map
    unordered_map<string, char> decodeMap;
    for (auto& p : tree.codeMap)
        decodeMap[p.second] = p.first;   // 编码 -> 字符

    ifstream fin("./huffman_code.txt");
    string bits;
    getline(fin, bits);          // ① 第一行就是码流
    fin.close();

    string cur, decoded;
    for (char b : bits) {
        cur += b;
        if (decodeMap.count(cur)) {      // 命中一个叶子
            decoded += decodeMap[cur];
            cur.clear();                 // 清空继续
        }
    }

    cout << "\n---------- result ----------\n";
    cout << "origin : " << target << '\n';
    cout << "after : " << decoded << '\n';

}



