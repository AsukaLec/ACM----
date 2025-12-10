#include "MyBinTree.hpp"

int main(){
// 2、基于用户输入的广义表串、前序扩展序列、中缀算术表达式，建立二叉树

//      广义表串：A(B(D,E(G,)),C(,F))

//      前序扩展：AB#D##C##

//      中缀表达式：a*(b+c)-d

//      建议：对输入做合法性判断，并给出相应提示

    BinaryTree<char> bt;
    char str[] = "A(B(D,E(G,)),C(,F))#";

    bt.createBinTree(str);
    bt.printBinTree(); cout << endl; //A (B (D ,E (G ,# )),C (# ,F ))

    bt.PreOrder_iter();  // A B D E C F
    bt.InOrder_iter();
    bt.PostOrder_iter();

    cout << endl;
    bt.PreOrder();
    bt.InOrder();
    bt.PostOrder(); 
    cout << endl;
    
    bt.LevelOrder(); cout << endl;
    //3、采用凹入表形式输出第2步创建的二叉树内容
    bt.Traverse(); 

    cout << "--- ---" << endl;
    //4. 基于用户输入的前序序列和中序序列，建立二叉树
    BinaryTree<char> bt1;
    char str1[] = "ABHFDECKG"; char str2[] = "HBDFAEKCG";
    int length = 10;
    bt1.VLR_LVR_create(str1, str2, length);
    //5、分别以前序、中序、后序、层次遍历方式，输出第4步创建的二叉树内容
    bt1.PreOrder(); 
    bt1.InOrder();
    bt1.PostOrder(); 
    bt1.LevelOrder();
    cout << endl;

    bt1.PreOrder_iter();
    bt1.InOrder_iter();
    bt1.PostOrder_iter();
    cout << endl;

    bt1.printBinTree();
    cout << endl;

    // 输出树的高度
    cout << bt1.Height() << " \n";
    cout << bt1.Height_nonRecursive() << " ";
}
