#include<iostream>
#include<ostream>
using namespace std;

class MyString {
    private:
        char* _str;
        long long* _next;

        size_t _length;
        size_t _capacity;

        // void getNext1() { 
        //     _next = new long long [_length] ;
        //     _next[0] = 0;
        //     size_t index = 1;
        //     size_t temp = 0;

        //     while ( index < _length) {
        //         if (_str[index] == _str[temp]) {
        //             _next[index] = _next[temp] + 1;
        //             ++index, ++temp;
        //         }
        //         else {
        //             while ((_str[temp] != _str[index])) {
        //                 temp = _next[temp - 1];
        //                 if (temp == 0) break;
        //             }

        //             if (temp == 0) {
        //                 if (_str[index] == _str[0]) _next[index] = 1;
        //                 else _next[index] = 0;
        //             } else _next[index] = _next[temp] + 1;

        //             ++index, ++temp;
        //         }
        //     }
        // }

            //递推生成Next数组
        void getNext() { //next数组的定义为：next[i]表示模式串A[0]至A[i]这个字串，使得前k个字符等于后k个字符的最大值，特别的k不能取i+i
            _next = new long long[_length];   
            _next[0] = 0;                     

            size_t index = 1;   
            size_t temp  = 0;  

            while (index < _length) {
                if (_str[index] == _str[temp]) {
                    _next[index] = temp + 1;
                    ++index; ++temp;
                }
                else { // 失配
                    if (temp == 0) {
                        _next[index] = 0;
                        ++index;          
                    } else {
                        temp = _next[temp - 1];  
                    }
                }
            }
        }


    public: 

        size_t getLength() {return _length;}
        size_t getCapacity() {return _capacity;}

        size_t strLen(char* str) {
            size_t len = 0;
            while(*str++) len++;
            return len;
        }

        MyString() {
            _capacity = 256;
            _str = new char[_capacity];
            _length = 0;
            _str[0] = '\0';
        }   

        MyString(const char* init) {
            _length = strLen((char*)init);
            _str = new char[_length + 1];
            _capacity = _length + 1;

            for (size_t i = 0; i < _length; ++i) _str[i] = init[i];
            _str[_length] = '\0';
            getNext();
        }

        MyString(const MyString& other) {
            _length = other._length;
            _capacity = other._capacity;
            _str = new char[_capacity];

            for(size_t i = 0; i < _length; i++) {
                _str[i] = other._str[i];
            }
            _str[_length] = '\0';
            getNext();
        }

        ~MyString() {
            delete[] _str;
            delete[] _next;
        }

        void expand () { // 扩容
            if (_length * 2 > _capacity) {
                _capacity *= 2;
                char* new_str = new char[_capacity];

                for (size_t i = 0; i < _length; i++) {
                    new_str[i] = _str[i];
                }
                new_str[_length] = '\0';
                delete[] _str;
                _str = new_str;
            }
        }

        void shrink() { // 缩容
            if (_length < _capacity / 4 && _capacity > 256) {
                _capacity /= 2;
                char* new_str = new char[_capacity];

                for (size_t i = 0; i < _length; i ++) {
                    new_str[i] = _str[i];
                } 
                new_str[_length] = '\0';
                delete[] _str;
                _str = new_str;
            }
        }

        MyString operator() (size_t pos, size_t len) {
            if (pos + len <= _length) {
                char* cur = new char[len + 1];
                for (size_t i = 0; i < len; ++i)   // 0..len-1
                    cur[i] = _str[pos + i];
                cur[len] = '\0';
                MyString cur_str = MyString(cur);
                delete[] cur;
                return cur_str;
            } else {
                throw out_of_range("SUbstring out of range");
            }
        }

        char operator[] (size_t index) {
            if (index < _length) {
                return _str[index];
            } else {
                throw out_of_range("Index out of range");
            }
        }

        MyString& operator= (const MyString& other) {
            if (this != &other) {
                delete[] _str;
                _length = other._length; _capacity = other._capacity; 
                _str = new char[_capacity];
                size_t cur_length = _length;

                while(cur_length --) {_str[cur_length] = other._str[cur_length];}
                _str[_length] = '\0';
            }
            return *this;
        }

        friend ostream& operator<< (ostream& out, const MyString& str) {
            // return out.write(str._str, str._length);
            for (size_t i = 0; i < str._length; i ++) {
                out << str._str[i];
            }
            return out;
        }

        bool operator == (const MyString& other) {
            if (_length != other._length) return false;
            for (size_t i = 0; i < _length; i++) {
                if (_str[i] != other._str[i]) return false;
            }
            return true;
        }

        bool operator != (const MyString& other) {
            return !(*this == other);
        }

        MyString& operator += (const MyString& other) {
            _length = _length + other._length;
            while(_length > _capacity) expand();

            for (size_t i = 0; i < other._length; i ++) {
                _str[this->_length - other._length + i] = other._str[i];
            }
            _str[_length] = '\0';  
            getNext();          
            return *this;
        }

        MyString& operator += (const char* other) {
            MyString tmp(other);
            *this += tmp;
            getNext();
            return *this;
        }

        long long  find_BF(MyString& pat, size_t pos = 0) { // 返回索引，从0开始
            size_t T_index = pos , P_index = 0;
            while (P_index < pat.getLength() && T_index < this->getLength()) {
                if (this->_str[T_index] == pat._str[P_index]) T_index ++, P_index ++;
                else T_index = T_index - P_index + 1, P_index = 0;
            }
            if (P_index == pat.getLength() && T_index <= this->getLength()) return static_cast<long long>(T_index - pat.getLength());
            return static_cast<long long>(-1); 
        }

        void showNext() {
            this->getNext();
            size_t i = 0; 
            while (i < _length) {
                cout << _next[i++] << " "; 
            }
        }

        long long find_KMP(const MyString& pat, size_t start = 0) const {
            if (pat._length == 0) return 0;                   
            if (pat._length > this->_length) return -1;       

            size_t i = start;   
            size_t j = 0;       

            while (i < this->_length) {                       
                if (j == pat._length)                         
                    return static_cast<long long>(i - j);
                if (j == 0 || this->_str[i] == pat._str[j]) { 
                    ++i;
                    ++j;
                } else {
                    j = pat._next[j - 1];                     
                }
            }
            return (j == pat._length) ? static_cast<long long>(i - j) : -1;
        }


};

int main() {
    MyString str("Hello, World!");
    cout << str.getLength() << endl; //13
    cout << str.getCapacity() << endl; // 14
    cout << str << endl; // "Hello, World!"
    // MyString sub = str(1, 5);
    // cout << sub << endl;
    // cout << str[7] << endl;

    MyString str1 ("Hello, World!");
    MyString str2("abcdefg");
    str += str2;
    cout << str.getLength() << endl; // 20
    cout << str.getCapacity() << endl; // 28
    cout << str << endl; //"Hello, World!abcdefg"

    MyString str_pat = "g";
    cout << str.find_BF(str_pat) << endl; // 19

    MyString text("ABABABCABABABC");
    MyString pat("ABABC");

    long long pos = text.find_KMP(pat);
    cout << pos << endl; 
}