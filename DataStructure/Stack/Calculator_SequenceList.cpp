#include "MY_STACK_BASE_ON_SEQLIST.hpp"

// 运算符优先级
map<char, int> precedence = {
    {'+', 1}, {'-', 1}, {'*', 2}, {'/', 2}
};

// 括号匹配判断
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

// 中缀表达式转后缀表达式
string infixToPostfix(const string& expr){
    Stack<char> opStack;
    string output;
    for (size_t i = 0; i < expr.size(); ++i) {
        char ch = expr[i];
        if (isdigit(ch)) {                    
            while (i < expr.size() && isdigit(expr[i]))
                output += expr[i++];          
            output += ' ';                    
            --i;                              
        }
        else if (ch == '(') {
            opStack.push(ch);
        }
        else if (ch == ')') {
            while (!opStack.empty() && opStack.top() != '(')
                output += opStack.pop(), output += ' ';
            if (!opStack.empty()) opStack.pop(); // 弹 '('
        }
        else if (precedence.count(ch)) {
            while (!opStack.empty() && opStack.top() != '(' &&
                   precedence[opStack.top()] >= precedence[ch]) {
                output += opStack.pop(), output += ' ';
            }
            opStack.push(ch);
        }
    }
    while (!opStack.empty())
        output += opStack.pop(), output += ' ';
    if (!output.empty() && output.back() == ' ')
        output.pop_back();           
    return output;
}


long long evalPostfix(const string& postfix)
{
    Stack<long long> numStack;
    for (size_t i = 0; i < postfix.size(); ++i) {
        char ch = postfix[i];
        if (isdigit(ch)) {       
            long long num = 0;
            while (i < postfix.size() && isdigit(postfix[i]))
                num = num * 10 + (postfix[i++] - '0');
            numStack.push(num);
            --i;
        }
        else if (precedence.count(ch)) {
            long long b = numStack.pop();
            long long a = numStack.pop();
            long long res = 0;
            switch (ch) {
                case '+': res = a + b; break;
                case '-': res = a - b; break;
                case '*': res = a * b; break;
                case '/':
                    if (b == 0) throw std::runtime_error("Divide by zero");
                    res = a / b;
                    break;
            }
            numStack.push(res);
        }
        else if (ch == ' ') continue;
    }
    return numStack.top();
}


int main() {
    string expr;
    cout << "Input: ";
    getline(cin, expr);

    if (!isParenthesesMatched(expr)) { // 括号匹配判断
        cout << "Mismatched Parentheses!" << endl;
        return 0;
    } else if (isParenthesesMatched(expr)) {
        cout << "Matched Parentheses!" << endl;
    }

    //中缀转后缀
    string postfix = infixToPostfix(expr);
    cout << "RPN: " << postfix << endl;
    //后缀表达式求值
    int result = evalPostfix(postfix);
    cout << "Result: " << result << endl;
    return 0;
}
