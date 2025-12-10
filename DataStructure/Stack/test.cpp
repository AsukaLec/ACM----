#include<iostream>
#include<sstream>
#include<string>
using namespace std;
int main() {
    string line;
    while(true){
        if (line == "q") break;

        getline(cin, line);
        stringstream ss(line);
        string tok;
        while(ss >> tok) cout << tok << endl;
    }

}