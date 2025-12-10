#include "MY_STACK_BASE_ON_LINKEDLIST.hpp"
#include <sstream>
#include <cctype>
using namespace std;


map<char,int> precedence = {
    {'+',1}, {'-',1},
    {'*',2}, {'/',2},
    {'~',3}          // 一元负号
};

/* 处理字符串为符号序列 */
vector<string> lexer(const string& s) {
    vector<string> res;
    stringstream ss(s);
    char c;
    while (ss >> c) {               
        if (c == '(' || c == ')' || c == '+' || c == '*' || c == '/') {
            res.push_back(string(1, c));
        }
        else if (c == '-') {        // 可能是二元减，也可能是一元负
            res.push_back(string(1, c));
        }
        else if (isdigit(c) || c == '.') {
            string num(1, c);
            while (ss.peek() == '.' || isdigit(ss.peek())) {
                num += ss.get();
            }
            res.push_back(num);
        }
        else {
            throw runtime_error("lexer: Unsopported character especially Chinese");
        }
    }
    return res;
}

/* 括号匹配 */
bool isParenthesesMatched(const string& expr) {
    Stack<char> s;
    for (char ch : expr) {
        if (ch == '(' || ch == '[' || ch == '{') {
            s.push(ch);
        } else if (ch == ')' || ch == ']' || ch == '}') {
            if (s.empty()) return false;
            char top = s.pop();
            if ((ch == ')' && top != '(') ||
                (ch == ']' && top != '[') ||
                (ch == '}' && top != '{')) {
                return false;
            }
        }
    }
    return s.empty();
}

/* 中缀转后缀 */
string infixToPostfix(const vector<string>& tk) {
    Stack<string> st;
    string out;
    auto appendNum = [&](const string& v) { out += v + ' '; };

    for (size_t i = 0; i < tk.size(); ++i) {
        const string& x = tk[i];
        char c = x[0];
        /* 数字 */
        if (isdigit(c) || (c == '-' && x.size() > 1)) {
            appendNum(x);
        }
        /* 左括号 */
        else if (c == '(') {
            st.push(x);
        }
        /* 右括号 */
        else if (c == ')') {
            while (!st.empty() && st.top() != "(")
                out += st.pop() + ' ';
            if (st.empty()) throw runtime_error("Mismatched )");
            st.pop(); // 弹 '('
        }
        /* 操作符 */
        else if (string("+-*/").find(c) != string::npos) {
            /* 判断一元 '-' */
            bool isUnary = (c == '-' &&
                (i == 0 || tk[i-1] == "(" || string("+-*/").find(tk[i-1][0]) != string::npos));
            string op = isUnary ? "~" : x;

            while (!st.empty() && st.top() != "(" &&
                   precedence[st.top()[0]] >= precedence[op[0]]) {
                out += st.pop() + ' ';
            }
            st.push(op);
        }
    }
    while (!st.empty()) out += st.pop() + ' ';
    if (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

/* 后缀求值 */
double evalPostfix(const string& postfix) {
    Stack<double> st;
    stringstream ss(postfix);
    string tok;
    while (ss >> tok) {
        char c = tok[0];
        if (isdigit(c) || (c == '-' && tok.size() > 1)) { // 数字
            st.push(stod(tok));
        }
        else if (tok == "~") {          // 一元负
            double a = st.pop();
            st.push(-a);
        }
        else if (string("+-*/").find(c) != string::npos) { // 二元运算
            double b = st.pop(), a = st.pop();
            switch (c) {
                case '+': st.push(a + b); break;
                case '-': st.push(a - b); break;
                case '*': st.push(a * b); break;
                case '/':
                    if (b == 0) throw runtime_error("Divide by zero");
                    st.push(a / b);
                    break;
            }
        }
        else throw runtime_error("Unknown token in eval");
    }
    if (st.stacksize() != 1) throw runtime_error("Illegal postfix");
    return st.top();
}

int main() {
    cout << "make sure there isn't Chinese Character in your input\n" 
        << "make sure you have separated numbers and operators with spaces\n" 
        << "use q to exit the programme" << endl;
    
    string line;
    cout << "Input: ";
    while (getline(cin, line)) {
        if (line == "q") break; //退出程序

        string s;
        for (char c : line) if (!isspace(c)) s += c;

        vector<string> tk = lexer(s);
        string postfix = infixToPostfix(tk);

        if (!isParenthesesMatched(s)) {
            cout << "Mismatched Parentheses!" << endl;
        } else {
            cout << "Matched Parentheses!" << endl;
            double ans = evalPostfix(postfix);
            cout << "RVN: " << postfix << endl;
            cout << "Result: " << ans << endl;
        }
        cout << "Input: ";
    }
    return 0;
}
