#include "MY_STACK_BASE_ON_LINKEDLIST.hpp"
using namespace std;

map<char,int> precedence = {
    {'+',1}, {'-',1},
    {'*',2}, {'/',2},
    {'~',3}         
};

struct Token {
    enum Type { NUM, OP, LP, RP } type;
    double num = 0;   
    char   op  = 0;   
};
using Tokens = vector<Token>;


Tokens lexer(const string& s) { // 把字符串拆成 token 序列
    Tokens tk;
    for (size_t i = 0; i < s.size(); ) {
        char c = s[i];
        if (c == ' ') { ++i; continue; }
        if (c == '(') { tk.push_back({Token::LP}); ++i; continue; }
        if (c == ')') { tk.push_back({Token::RP}); ++i; continue; }
        if (string("+-*/").find(c) != string::npos) {
            /* 判断一元 '-' */
            bool isUnary = (c == '-' && 
                (tk.empty() || tk.back().type == Token::OP || tk.back().type == Token::LP));
            if (isUnary) {
                tk.push_back({Token::OP, 0, '~'});
                ++i;
            } else {
                tk.push_back({Token::OP, 0, c});
                ++i;
            }
            continue;
        }
        if (isdigit(c) || c == '.') {
            size_t j = i;
            while (j < s.size() && (isdigit(s[j]) || s[j] == '.')) ++j;
            double v = stod(s.substr(i, j - i));
            tk.push_back({Token::NUM, v, 0});
            i = j;
            continue;
        }
        throw runtime_error("Unknown character in lexer");
    }
    return tk;
}

bool isParenthesesMatched(const string& expr) { // 括号匹配判断
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

string infixTokensToPostfix(const Tokens& tk) { //中缀转后缀
    Stack<char> st;          
    string out;             
    auto appendNum = [&](double v) {
        out += to_string(v) + ' ';
    };
    for (const auto& x : tk) {
        if (x.type == Token::NUM) {
            appendNum(x.num);
        }
        else if (x.type == Token::LP) {
            st.push('(');
        }
        else if (x.type == Token::RP) {
            while (!st.empty() && st.top() != '(')
                out += string(1, st.pop()) + ' ';
            if (st.empty()) throw runtime_error("Mismatched )");
            st.pop(); // 弹 '('
        }
        else if (x.type == Token::OP) {
            char op = x.op;
            if (op == '~') {                // 一元负号直接压
                st.push(op);
            } else {
                while (!st.empty() && st.top() != '(' &&
                       precedence[st.top()] >= precedence[op]) {
                    out += string(1, st.pop()) + ' ';
                }
                st.push(op);
            }
        }
    }
    while (!st.empty()) {
        out += string(1, st.pop()) + ' ';
    }
    if (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

double evalPostfix(const string& postfix) { //后缀求值
    Stack<double> st;
    stringstream ss(postfix);
    string tok;
    while (ss >> tok) {
        if (isdigit(tok[0]) || (tok.size() > 1 && tok[0] == '-' && isdigit(tok[1]))) {
            st.push(stod(tok));
        } else if (tok.size() == 1 && precedence.count(tok[0])) {
            char op = tok[0];
            if (op == '~') {
                double a = st.pop();
                st.push(-a);
            } else {
                double b = st.pop();
                double a = st.pop();
                switch (op) {
                    case '+': st.push(a + b); break;
                    case '-': st.push(a - b); break;
                    case '*': st.push(a * b); break;
                    case '/':
                        if (b == 0) throw runtime_error("Divide by zero");
                        st.push(a / b);
                        break;
                }
            }
        } else {
            throw runtime_error("Unknown token in eval");
        }
    }
    if (st.stacksize() != 1) throw runtime_error("Illegal postfix");
    return st.top();
}

int main() {
    string line;
    cout << "Input: ";
    while (getline(cin, line)) {
        if (line == "q") break;

        string s;
        for (char c : line) if (!isspace(c)) s += c;

        Tokens tk = lexer(s);
        string postfix = infixTokensToPostfix(tk);

        if (!isParenthesesMatched(postfix)) { //检查括号匹配
            cout << "Mismatched Parentheses!" << endl;
            return 0;
        } else if (isParenthesesMatched(postfix)) {
            cout << "Matched Parentheses!" << endl;
        }

        double ans = evalPostfix(postfix);

        cout << "RVN: " << postfix << endl; // 后缀表达式
        cout << "Result : " << ans << endl; // 计算结果

        
        cout << "Input: ";
    }
    return 0;
}
