* [二维数组中的查找](https://www.nowcoder.com/practice/abc3fe2ce8e146608e868a70efebf62e?tpId=13&tqId=11154&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

  ```
  public class Solution {
      public boolean Find(int target, int [][] array) {
          int row = array.length;
          if(row == 0) {
              return false;
          }
          int col = array[0].length;
          if(col == 0) {
              return false;
          }       
          // 最右上角的位置
          int r = 0, c = col - 1;
          // 从右上开始向右下对比
          while(r < row && c >= 0) {
              if(target == array[r][c]) {
                  return true;
              } else if(target > array[r][c]) {
                  // 说明目标值在[r][c]的下边
                  ++r;
              } else {
                  // 说明目标值在[r][c]的左边
                  --c;
              }
          }        
          return false;
      }
  }
  ```

* [替换空格](nowcoder.com/practice/0e26e5551f2b489b9f58bc83aa4b6c68?tpId=13&tqId=11155&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。

    ```
    public class Solution {
    
        public String replaceSpace (String s) {
            // write code here
            StringBuilder builder = new StringBuilder();
            for(char c : s.toCharArray()) {
                builder.append(c == ' ' ? "%20" : c);
            }
            return builder.toString();
        }
    }
    ```

* [从头到尾打印链表](https://www.nowcoder.com/practice/d0267f7f55b3412ba93bd35cfa8e8035?tpId=13&tqId=11156&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 输入一个链表，按链表从尾到头的顺序返回一个ArrayList。

    ```
    import java.util.ArrayList;
    public class Solution {
        ArrayList<Integer> res = new ArrayList<>();    
        public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
            if(listNode != null) {
                this.printListFromTailToHead(listNode.next);
                res.add(listNode.val);
            }
            return res;
        }
    }
    ```

* [重建二叉树](https://www.nowcoder.com/practice/8a19cbe657394eeaac2f6ea9b0f6fcf6?tpId=13&tqId=11157&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。

    ```
    import java.util.*;
    public class Solution {
        public TreeNode reConstructBinaryTree(int [] pre,int [] in) {
            if(pre.length == 0 || in.length == 0) {
                return null;
            }
            // 前序遍历头元素为当前树的根节点
            TreeNode root = new TreeNode(pre[0]);
            for(int i = 0; i < in.length; i++) {
                if(pre[0] == in[i]) {
                    // copyOfRange是左闭右开(]   
                    root.left = reConstructBinaryTree(Arrays.copyOfRange(pre, 1, i+1), Arrays.copyOfRange(in, 0, i+1));
                    root.right = reConstructBinaryTree(Arrays.copyOfRange(pre, i+1, pre.length), Arrays.copyOfRange(in, i+1, in.length));
                    // 结束本次递归
                    break;
                }
            }
            return root;        
        }
    }
    ```

* [两个栈实现队列](https://www.nowcoder.com/practice/54275ddae22f475981afa2244dd448c6?tpId=13&tqId=11158&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。

    ```
    import java.util.Stack;
    
    public class Solution {
        Stack<Integer> stack1 = new Stack<Integer>();
        Stack<Integer> stack2 = new Stack<Integer>();
        
        public void push(int node) {
            stack1.push(node);
        }
        
        public int pop() {
            if(stack2.size() == 0) {
                while(stack1.size() != 0) {
                    stack2.push(stack1.pop());           
                }
            }
            return stack2.pop();
        }
    }
    ```

* [旋转数组的最小数字](https://www.nowcoder.com/practice/9f3231a991af4f55b95579b44b7a01ba?tpId=13&tqId=11159&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
    输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
    NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。

    ```
    import java.util.ArrayList;
    public class Solution {
        public int minNumberInRotateArray(int [] array) {
            int len = array.length;
            if(len == 0) {
                return 0;
            }
            
            int left = 0;
            int right = len-1;
            int mid = 0;
            // 二分法，从数组中间元素开始向右二分
            while(left < right) {
                if(array[left] < array[right]) {
                    return array[left];
                }
                
                mid = (left + right) / 2;
                // mid小于right，则res在左边
                if(array[mid] < array[right]) {
                    right = mid;
                // mid大于right，则res在右边
                } else if(array[mid] > array[left]) {
                    left = mid + 1;
                } else {
                    left++;
                }
            }
            return array[left];        
         }
    }
    ```

* [斐波那契数列](https://www.nowcoder.com/practice/c6c7742f5ba7442aada113136ddea0c3?tpId=13&tqId=11160&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0，第1项是1）。

    n\leq 39*n*≤39

    ```
    public class Solution {
        public int Fibonacci(int n) {
            if(n == 0 || n == 1) {
                return n;
            } 
            int a = 0, b = 1, sum = 0;
            while(n-- > 1) {
                sum = a + b;
                a = b;
                b = sum;
            }
            return sum;
        }
    }
    ```

* [跳台阶](https://www.nowcoder.com/practice/8c82a5b80378478f9484d87d1c5f12a4?tpId=13&tqId=11161&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。

    ```
    public class Solution {
        public int JumpFloor(int target) {
            if(target <= 1) {
                return target;
            }
            int a = 1, b = 1, c = 0;
            // 从第2阶开始算起，总共需要经历 target-1 次
            while(target-- > 1) {
                c = a + b;
                a = b;
                b = c;
            }
            return c;
        }
    }
    ```

* [跳台阶扩展问题](https://www.nowcoder.com/practice/22243d016f6b47f2a6928b4313c85387?tpId=13&tqId=11162&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。

  ```
  public class Solution {
      public int JumpFloorII(int target) {
          if(target == 1 || target == 0) {
              return 1;
          }
          int a = 1;
          int b = 0;
          
          for (int i = 2; i <= target; ++i) {
              b = a << 1;
              a = b;
          }
          return b;
      }
  }
  ```

* [矩形覆盖](https://www.nowcoder.com/practice/72a5a919508a4251859fb2cfb987a0e6?tpId=13&tqId=11163&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？

    ```
    public class Solution {
        public int rectCover(int target) {
            if(target <= 2) {
                return target;
            }
            int a = 1, b = 2, c = 0;
            // 从3遍历到target
            while(target-- >= 3) {
                c = a + b;
                a = b;
                b = c;
            }
            return c;
        }
    }
    ```

* [二进制中1的个数](https://www.nowcoder.com/practice/8ee967e43c2c4ec193b040ea7fbb10b8?tpId=13&tqId=11164&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 输入一个整数，输出该数32位二进制表示中1的个数。其中负数用补码表示。

    ```
    public class Solution {
        public int NumberOf1(int n) {
            int cnt = 0;
            while(n != 0) {
                n = n & (n - 1);
                cnt++;
            }
            return cnt;
        }
    }
    ```

* [数值的整数次方](https://www.nowcoder.com/practice/1a834e5e3e1a4b7ba251417554e07c00?tpId=13&tqId=11165&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。

    保证base和exponent不同时为0

    ```
    public class Solution {
        public double Power(double base, int exponent) {
            if(exponent < 0) {
                exponent = -exponent;
                base = 1 / base;
            }
            // 记录 x^0, x^1, x^2
            double x = base;
            double res = 1.0;
            // 指数为二进制位
            while(exponent > 0) {
                // 如果二进制最小位为1，则需要加入res
                if((exponent & 1) == 1) {
                    res *= x;
                }
                x *= x; // x^1 -> x^2
                exponent = exponent >> 1;
            }
            return res;
        }
    }
    ```

* [调整数组顺序使奇数位于偶数前面](https://www.nowcoder.com/practice/ef1f53ef31ca408cada5093c8780f44b?tpId=13&tqId=11166&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。

    ```
    public class Solution {
        public int[] reOrderArray (int[] array) {
            // write code here
            // 奇数odd，偶数even
            LinkedList<Integer> odd = new LinkedList<Integer>();
            LinkedList<Integer> even = new LinkedList<Integer>();
              
            for(int num : array) {
                if(num % 2 == 0) {
                    even.add(num);
                } else {
                    odd.add(num);
                }
            }
            for(int i = 0; i < array.length; i++) {
                if(!odd.isEmpty()) {
                    array[i] = odd.poll();
                } else {
                    array[i] = even.poll();
                }
            }
            return array;
        }
    }
    ```

* [链表中倒数第K个节点](https://www.nowcoder.com/practice/886370fe658f41b498d40fb34ae76ff9?tpId=13&tqId=11167&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 输入一个链表，输出该链表中倒数第k个结点。

    如果该链表长度小于k，请返回空。

    ```
    public class Solution {
        public ListNode FindKthToTail (ListNode pHead, int k) {
            if(pHead == null || k < 1) {
                return null;
            }
            // write code here
            ListNode slow = pHead, fast = pHead;
            // 快指针比慢指针先走k步
            while(k-- > 0) {
                // 说明ListNode长度小于k
                if(fast == null) {
                    return null;
                }
                fast = fast.next;
            }
            while(fast != null) {
                slow = slow.next;
                fast = fast.next;
            }
            return slow;
        }
    }
    ```

* [反转链表](https://www.nowcoder.com/practice/75e878df47f24fdc9dc3e400ec6058ca?tpId=13&tqId=11168&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 输入一个链表，反转链表后，输出新链表的表头。

    ```
    public class Solution {
        public ListNode ReverseList(ListNode head) {
            if(head == null || head.next == null) {
                return head;
            }
            ListNode prev = null, next = null;
            while(head != null) {
                // 保存后面链表
                next = head.next;
                // 当前节点head指针反转
                head.next = prev;
                // 下一次循环，prev在上一个节点，next在那一次的当前节点
                prev = head;
                head = next;
            }
            return prev;
        }
    }
    ```

* [合并两个排序的链表](https://www.nowcoder.com/practice/d8b6b4358f774294a89de2a6ac4d9337?tpId=13&tqId=11169&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。

    ```
    public class Solution {
        // 递归函数功能：合并两个单链表，返回两个单链表头结点值小的那个节点。
        public ListNode Merge(ListNode list1,ListNode list2) {
            if(list1 == null) {
                return list2;
            }         
            if(list2 == null) {
                return list1;
            }
            // 如果list1第一个元素大于list2首元素，所有递归结束后返回list1
            if(list1.val <= list2.val) {
                list1.next = Merge(list1.next, list2);
                return list1;
            } else {
                list2.next = Merge(list1, list2.next);
                return list2;
            }
        }
    }
    ```

* [树的子结构](https://www.nowcoder.com/practice/6e196c44c7004d15b1610b9afca8bd88?tpId=13&tqId=11170&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）

    ```
    public class Solution {
        public boolean HasSubtree(TreeNode root1,TreeNode root2) {
            if ( root2 == null || root1 == null){
                return false;
            }
            // 递归root1所有可能存在子树为root2的情况
            return isSubTree(root1, root2) || HasSubtree(root1.left, root2) || HasSubtree(root1.right, root2); 
        }
        
        private boolean isSubTree(TreeNode root1, TreeNode root2) {
            // 当前树节点为null，表示root2已经递归到叶子节点，当前子树是root1的子树 
            if(root2 == null) {
                return true;
            }
            if(root1 == null || root1.val != root2.val) {
                return false;
            }
            // 当前树节点相等，需要递归, 即判断root2的左右子树是否也与root1左右子树相等
            return isSubTree(root1.left, root2.left) && isSubTree(root1.right, root2.right);
        }
    }
    ```

* [二叉树的镜像](https://www.nowcoder.com/practice/a9d0ecbacef9410ca97463e4a5c83be7?tpId=13&tqId=11171&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 操作给定的二叉树，将其变换为源二叉树的镜像。

    ```
    public class Solution {
        /**
         * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
         *
         * 
         * @param pRoot TreeNode类 
         * @return TreeNode类
         */
        public TreeNode Mirror (TreeNode pRoot) {
            // write code here
            if(pRoot == null) {
                return null;
            }
            // 左子树节点
            TreeNode left = pRoot.left;
            // 每次遍历都交换当前节点的左右子节点
            pRoot.left = pRoot.right;
            pRoot.right  = left;
            // 前序遍历，递归左右子树，
            pRoot.left = Mirror(pRoot.left);
            pRoot.right = Mirror(pRoot.right);
            return pRoot;
        }
    }
    ```

* [顺时针打印矩阵](https://www.nowcoder.com/practice/9b4c81a02cd34f76be2659fa0d54342a?tpId=13&tqId=11172&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.

    ```
    import java.util.ArrayList;
    public class Solution {
        public ArrayList<Integer> printMatrix(int [][] matrix) {
            ArrayList<Integer> res = new ArrayList<>();
            if(matrix.length == 0 || matrix[0].length == 0) {
                return res;            
            }
            int up = 0, down = matrix.length - 1, left = 0, right = matrix[0].length - 1;
            int i = 0;
            while(true) {
                // 向右走
                for(i = left; i <= right; i++) {
                    res.add(matrix[up][i]);
                }
                // 当前行走完，更新up
                up++;
                if(up > down) {
                    break;
                }
                // 向下走
                for(i = up; i <= down; i++) {
                    res.add(matrix[i][right]);
                }
                // 当前列走完，更新right
                right--;
                if(left > right) {
                    break;
                }
                // 向左走
                for(i = right; i >= left; i--) {
                    res.add(matrix[down][i]);
                }
                // 下边界缩减
                down--;
                if(up > down) {
                    break;
                }
                // 向上走
                for(i = down; i >= up; i--) {
                    res.add(matrix[i][left]);
                }
                // 左边界向右
                left++;
                if(left > right) {
                    break;
                }
            }
            return res;       
        }
    }
    ```

* [包含main函数的栈](https://www.nowcoder.com/practice/4c776177d2c04c2494f2555c9fcc1e49?tpId=13&tqId=11173&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。

    ```
    import java.util.Stack;
    
    public class Solution {
        // 从栈顶到栈底数字越来越大
        static Stack<Integer> stack = new Stack<>();
        static Integer min = Integer.MAX_VALUE;
        public void push(int node) {
            if(stack.empty()) {
                min = node;
                stack.push(node);
            } else {
                if(node <= min) {
                    //在push更小的值时需要保留在此值之前的最小值
                    stack.push(min);
                    min = node;
                }
                stack.push(node);
            }
        }
        
        public void pop() {
            if(stack.size() == 0) {
                return;
            }
            if(min == stack.peek()) {
                if(stack.size() > 1) {
                    stack.pop();
                    // 最小值被获取，重新买更新最小值
                    min = stack.peek();
                } else {
                    // 栈中只有一个元素刚好等于min。 
                    min = Integer.MAX_VALUE;
                }
            }
            stack.pop();
        }
        
        public int top() {
            return stack.peek();
        }
        
        public int min() {
            return min;
        }
    }
    ```

* [栈的压入、弹出序列](https://www.nowcoder.com/practice/d77d11405cc7470d82554cb392585106?tpId=13&tqId=11174&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）

    ```
    import java.util.ArrayList;
    import java.util.Stack;
    
    public class Solution {
        public boolean IsPopOrder(int [] pushA,int [] popA) {
            if(pushA.length == 0 || popA.length == 0 || pushA.length != popA.length) {
                return false;
            }
            Stack<Integer> stack = new Stack<>();
            int idx = 0;
            for(int i = 0; i < pushA.length; i++) {
                stack.push(pushA[i]);
                while(!stack.isEmpty() && popA[idx] == stack.peek()) {
                    stack.pop();
                    idx++;
                }
            }
            return stack.isEmpty();
        }
    }
    ```

* [从上到下打印二叉树](https://www.nowcoder.com/practice/7fe2212963db4790b57431d9ed259701?tpId=13&tqId=11175&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 从上往下打印出二叉树的每个节点，同层节点从左至右打印。

    ```
    public class Solution {
        public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
            ArrayList<Integer> res = new ArrayList<>();
            if(root == null) {
                return res;
            }
            // 存放TreeNode
            Queue<TreeNode> treeQueue = new LinkedList<>();
            treeQueue.add(root);
            // BFS
            while(treeQueue.size() != 0) {
                // 取出队头元素
                TreeNode node = treeQueue.poll();
                if(node.left != null) {
                    treeQueue.add(node.left);
                }
                if(node.right != null) {
                    treeQueue.add(node.right);
                }
                res.add(node.val);
            }
            return res;
        }
    }
    ```

* [二叉搜索树的后序遍历序列](https://www.nowcoder.com/practice/a861533d45854474ac791d90e447bafd?tpId=13&tqId=11176&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则返回true,否则返回false。假设输入的数组的任意两个数字都互不相同。（ps：我们约定空树不是二叉搜素树）

    ```
    public class Solution {
        public boolean VerifySquenceOfBST(int [] sequence) {
            if(sequence.length == 0) {
                return false;
            }
            return check(sequence, 0, sequence.length - 1);
        }
            // 递归函数功能：node的left到right的树是否满足BST
        public boolean check(int[] node, int left, int right) {
            if(left >= right) {
                return true;
            }        
            // BST后序遍历数组的尾元素为当前树的根节点
            // root在当前递归里当做左右子树的分界点
            int root = right;
            // 从右到左找到第一个小于当前子树根节点，root表示左右子树的分界点，左子树的最大节点
            while(root > left && node[root] >= node[right]) {
                root--;
            }
            // 此时root右边的节点都小于根节点node[right]， 因此只需要判断左子树
            // 遍历左子树
            for(int i = root - 1; i >= left; --i) {
                // 如果左子树有一个节点大于当前子树根节点node[right], 则不满足BST
                if(node[i] > node[right]) {
                    return false;
                }
            }
            // 递归, 此时root为左子树最大节点, left和right在当前递归里不变
            return check(node, left, root) && check(node, root+1, right - 1);
        }
    }
    ```

* [二叉树中和为某一值的路径](https://www.nowcoder.com/practice/b736e784e3e34731af99065031301bca?tpId=13&tqId=11177&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 输入一颗二叉树的根节点和一个整数，按字典序打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。

    ```
    public class Solution {
        
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();    
        public ArrayList<ArrayList<Integer>> FindPath(TreeNode root,int target) {
            if(root == null) {
                return res;
            }
            dfs(root, target);
            return res;
        }
        
        ArrayList<Integer> list = new ArrayList<>();
        void dfs(TreeNode root, int target) {
            if(root == null) {
                return;
            }
            list.add(root.val);
            target -= root.val;
            // 达到目标值并且当前已经是叶子节点（题目要求）
            if(target == 0 && root.left == null && root.right == null) {
                res.add(new ArrayList<Integer>(list));
            }
            dfs(root.left, target);
            dfs(root.right, target);
            // 回溯
            list.remove(list.size() -1);
        }
    }
    ```

* [复杂链表的复制](https://www.nowcoder.com/practice/f836b2c43afc4b35ad6adc41ec941dba?tpId=13&tqId=11178&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * ·输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针random指向一个随机节点），请对此链表进行深拷贝，并返回拷贝后的头结点。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）

    ```
    public class Solution {
        public RandomListNode Clone(RandomListNode pHead) {
            if(pHead == null) {
                return null;
            }
            
            RandomListNode node = pHead;
            // 复制每个节点
            while(node != null) {
                RandomListNode cloneNode = new RandomListNode(node.label);
                // 新的复制的节点在老节点后面
                cloneNode.next = node.next;
                node.next = cloneNode;
                node = cloneNode.next;   
            }        
            // 复制完毕后，·重新遍历链表，复制旧节点的随机指针到每个新节点
            node = pHead;
            while(node != null) {
                // 新节点的随机指针指向旧节点的随机指针后面的节点
                // 新节点的随机指针赋值
                node.next.random = node.random == null ? null : node.random.next;
                node = node.next.next;
            }
            // 拆分出新链表
            node = pHead;
            RandomListNode pNewHead = pHead.next;
            while(node != null) {
                // 新链表节点
                RandomListNode nextNode = node.next;
                node.next = nextNode.next;
                nextNode.next = nextNode.next == null ? null : nextNode.next.next;
                node = node.next;
            }
            return pNewHead;
        }
    }
    ```

* [二叉搜索树与双向链表](https://www.nowcoder.com/practice/947f6eb80d944a84850b0538bf0ec3a5?tpId=13&tqId=11179&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。

    ```
    public class Solution {
        // 记录左子树的最后一个节点
        TreeNode leftLast = null;
        // 递归函数功能：返回子树的叶子节点，同时记录叶子节点为leftLast
        public TreeNode Convert(TreeNode pRootOfTree) {
            // base case
            if(pRootOfTree == null) {
                return null;
            }
            // 递归到叶子节点
            if(pRootOfTree.left == null && pRootOfTree.right == null) {
                // 每次记录当前子树的叶子节点，从最左叶子节点记录到最右叶子节点
                leftLast = pRootOfTree;
                return pRootOfTree;
            }
            // 递归1：构建左子树的双链表
            TreeNode left = Convert(pRootOfTree.left);
            if(left != null) {
                // 每次递归将节点加入双链表，从最左叶子节点开始构建
                // 因为要按顺序构建双链表，而二叉树left<root<right，最左叶子节点最小
                leftLast.right = pRootOfTree;; // 左叶子节点->当前节点->右叶子节点
                pRootOfTree.left = leftLast; 
            }
            // 当前根节点只包含左子树时，说明该根节点为最后一个节点
            leftLast = pRootOfTree;
            // 递归2：从最左边叶子节点开始递归右子树，构建右子树的双链表，递归直到right为最右叶子节点
            TreeNode right = Convert(pRootOfTree.right);
            if(right != null) {
                // 比right小的节点都放在right的子树
                // 相当于双链表的新节点的头指针指向双链表
                right.left = pRootOfTree;
                // 双链表的左右指针指向
                // 相当于当前双链表尾指针指向下一个比当前节点大的右节点right
                pRootOfTree.right = right;
            }
            // left为null只有一种情况：原来的二叉搜索树没有左子树
            return left != null ? left : pRootOfTree;
        }
    }
    ```

* [字符串的排列](https://www.nowcoder.com/practice/fe6b651b66ae47d7acce78ffdd9a96c7?tpId=13&tqId=11180&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则按字典序打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。

    ```
    import java.util.*;
    public class Solution {
        ArrayList<String> res = new ArrayList<>();
        public ArrayList<String> Permutation(String str) {
            if(str.length() == 0) {
                return res;
            }
            backtrack(str.toCharArray(), 0);
            // 题目要求按字典顺序输出
            Collections.sort(res);
            return res;
        }
        
        // 递归函数功能：以arr[idx]为开头的字符串结果为全排列的其中一种结果
        void backtrack(char[] arr, int idx) {
            // base case 遍历数组arr完毕，退出递归
            if(idx == arr.length - 1) {   
                // 不添加重复的结果
                if(!res.contains(String.valueOf(arr))) {
                    // 添加路径
                    res.add(String.valueOf(arr));
                }
                return;
            } 
            // 选择列表
            for(int i = idx; i < arr.length; i++) {
                // 交换两个元素，如果交换后相等则不添加结果
                swap(arr, i, idx);
                // 递归，继续递归决策树
                backtrack(arr, idx + 1);
                // 当前递归函数执行完毕后交换回来，回到原来的状态，为了下一次for循环的顺序
                swap(arr, i, idx);
            }
        }
        
        void swap(char[] arr, int i, int j) {
            char temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    ```

* [数组中出现次数超过一半的数字](https://www.nowcoder.com/practice/e8a1b01a2df14cb2b228b30ee6a92163?tpId=13&tqId=11181&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。

    ```
    public class Solution {
        public int MoreThanHalfNum_Solution(int [] array) {
            if(array.length == 0) {
                return 0;
            }
            int cnt = 1, num = array[0];
            // 找出数组中出现次数最多的元素num
            for(int i = 1; i < array.length; i++) {
                if(array[i] == num) {
                    cnt++;
                } else {
                    cnt--;
                }
                if(cnt == 0) {
                    num = array[i];
                    cnt = 1;
                }
            }
            cnt = 0;
            // 判断出现次数最多的元素num个数是否超过数组长度一半
            for(int i = 0; i < array.length; i++) {
                if(num == array[i]) {
                    cnt++;
                }
            }
            if(cnt * 2 > array.length) {
                return num;
            } else {
                return 0;
            }
        }
    }
    ```

* [最小的K个数](https://www.nowcoder.com/practice/6a296eb82cf844ca8539b57c23e6e9bf?tpId=13&tqId=11182&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 给定一个数组，找出其中最小的K个数。例如数组元素是4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4。如果K>数组的长度，那么返回一个空的数组

    ```
    import java.util.ArrayList;
    
    public class Solution {
        public ArrayList<Integer> GetLeastNumbers_Solution(int [] input, int k) {
            ArrayList<Integer> res = new ArrayList<>();
            if(k > input.length) {
                return res;
            }
            quickSort(input, 0, input.length - 1);
            for(int i = 0; i < k; i++) {
                res.add(input[i]);
            }        
            return res;
        }
        
        // 快排
        void quickSort(int[] arr, int left, int right) {
            if(left < right) {
                // 每次数组的最左边元素为基准数
                int lo = left, hi = right, base = arr[left];
                while(lo < hi) {
                    // 从右到左找到第一个小于base的数
                    while(lo < hi && arr[hi] >= base) {
                        hi--;
                    }
                    // 找的后填补到base基准数的位置
                    if(lo < hi) {
                        arr[lo++] = arr[hi];
                    }
                    // 从左到右找出第一个大于base的数
                    while(lo < hi && arr[lo] <= base) {
                        lo++;
                    }
                    // 填补上一个hi的位置，填补到最后的lo下标位置留给base填补
                    if(lo < hi) {
                        arr[hi--] = arr[lo];
                    }
                }
                // base基准数填补最后的lo位置
                arr[lo] = base;
                // 递归对base位置的左右子数组进行快排
                quickSort(arr, left, lo - 1);
                quickSort(arr, lo + 1, right);
            }
        }
    }
    ```

* [连续子数组的最大和](https://www.nowcoder.com/practice/459bd355da1549fa8a49e350bf3df484?tpId=13&tqId=11183&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 输入一个整型数组，数组里有正数也有负数。数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。要求时间复杂度为 O(n).

    ```
    public class Solution {
        public int FindGreatestSumOfSubArray(int[] array) {
            if(array.length == 0) {
                return 0;
            }        
            int res = array[0], max = array[0];
            for(int i = 1; i < array.length; i++) {
                // max表示当前以第i个数组为末尾的连续子数组最大和
                max = Math.max(array[i], max + array[i]);
                res = Math.max(max, res);
            }
            return res;
        }
    }
    ```

* [整数中1出现的次数](https://www.nowcoder.com/practice/bd7f978302044eee894445e244c7eee6?tpId=13&tqId=11184&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。

    ```
    public class Solution {
        public int NumberOf1Between1AndN_Solution(int n) {
            // cur从个位开始，低位数low始终在cur右边，高位数high总在cur坐边
            // 如n=2304，high=23、cur=0，low=4，digit=10
            int low = 0, cur = n % 10, high = n / 10;
            // digit表示当前cur的位数，个位为1，十位为2
            int digit = 1, res = 0;
            // 当high与cur同时为0时，说明已经越过最高位，跳出
            while(high != 0 || cur != 0) {
                // 情况1：当前位为0，count1 = high*digit
                if(cur == 0) {
                    res += high * digit;
                } else if(cur == 1) {
                    // 情况2：当前位为1， count1 = high * digit + (low+1)
                    res += high * digit + (low + 1);
                } else {
                    // 情况3：当前位大于1，count1 = high * digit + digit
                    // 多出来的digit为当前位cur为1时的情况
                    res += high * digit + digit; 
                }
                low += cur * digit;
                digit *= 10;
                cur = high % 10;
                high /= 10;
            }
            return res;
        }
    }
    ```

* [把数组排成最小的数](https://www.nowcoder.com/practice/8fecd3f8ba334add803bf2a06af1b993?tpId=13&tqId=11185&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。

    ```
    import java.util.*;
    
    public class Solution {
        public String PrintMinNumber(int [] numbers) {
            if(numbers.length == 0) {
                return "";
            }
            String res = "";
            List<Integer> list = new ArrayList<>();
            for(int num : numbers) {
                list.add(num);
            }
            // 自定义排序规则
            Collections.sort(list, new Comparator<Integer>() {
                public int compare(Integer num1, Integer num2) {
                    String s1 = num1 + "" + num2;
                    String s2 = num2 + "" + num1;
                    // 比较s1和s2字符编码之和的大小，list按升序从小到大排列
                    return s1.compareTo(s2);
                }
            });
            for(int num : list) {
                res += num;
            }
            return res;
        }
    }
    ```

* [丑数](https://www.nowcoder.com/practice/6aa9e04fc3794f68acf8778237ba065b?tpId=13&tqId=11186&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * ·把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。

    ```
    public class Solution {
        public int GetUglyNumber_Solution(int index) {
            if(index < 7) {
                return index;
            }
            // 三个指针分别用于指向三个队列
            // res表示每次从三个队列中从队头选出最小的数，1为最小丑数
            int p2 = 0, p3 = 0, p5 = 0, res = 1;
            // 用于保存顺序的丑树
            int[] arr = new int[index + 1];
            // 先保存最小丑数1
            arr[0] = res;
            // 从第2个树开始查找，因为第一个丑数已经设定为 1
            int idx = 1;
            // 只需要查找 index-1 次即可，因为第1个丑数为1已经找出
            // 加入要找到第 7 个丑数，只需要查找6次即可
            while(--index > 0) {
                // 从三个指针中选出乘后的最小的丑数
                res = Math.min(arr[p2] * 2, Math.min(arr[p3] * 3, arr[p5] * 5));
                // 更新指针
                if(arr[p2] * 2 == res) {
                    p2++;
                } 
                if(arr[p3] * 3 == res) {
                    p3++;
                }
                if(arr[p5] * 5 == res) {
                    p5++;
                }
                // 每次添加当次操作的最小丑数
                arr[idx++] = res;
            }
            return res;
        }
    }
    ```

* [第一个只出现一次的字符位置](https://www.nowcoder.com/practice/1c82e8cf713b4bbeb2a5b31cf5b0417c?tpId=13&tqId=11187&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则返回 -1（需要区分大小写）.（从0开始计数）

    ```
    import java.util.HashMap;
    import java.util.Map;
    
    public class Solution {
        public int FirstNotRepeatingChar(String str) {
            Map<Character, Integer> map = new HashMap<>();
            for(int i = 0; i < str.length(); i++) {
                map.put(str.charAt(i), map.getOrDefault(str.charAt(i), 0) + 1);
            }
            for(int i = 0; i < str.length(); i++) {
                if(map.get(str.charAt(i)) == 1) {
                    // 直接返回结果 
                    return i;
                }
            }
            return -1;
        }
    }
    ```

* [数组中逆序对](https://www.nowcoder.com/practice/96bd6684e04a44eb80e6a68efc0ec6c5?tpId=13&tqId=11188&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007

    ```
    public class Solution {
        int res = 0;
    int[] nums;
        public int InversePairs(int [] array) {
          if(array.length == 0) {
                return 0;
            }
            nums = new int[array.length];
            mergerSort(array, 0, array.length - 1);
            return res; 
        }
        
        void mergerSort(int[] arr, int left, int right) {
            // 递归到只有一个元素
            if(left >= right) {
                return;
            }
            int mid = left + ((right - left) >> 1);
            // 归
            mergerSort(arr, left, mid);
            mergerSort(arr, mid + 1, right);
            // lo和hi分别指向两个子数组的第一个元素, idx为新数组的下标
            int lo = left, hi = mid + 1, idx = 0;
            // 比较
            while(lo <= mid && hi <= right) {
                if(arr[lo] <= arr[hi]) {
                    nums[idx++] = arr[lo++]; 
                } else {
                    // arr[lo] > arr[hi]，即前面的大于后面的，此时就需要计数
                    // 因为此时的左右两个子树组是内部有序的，
                    // 所以arr[lo]>arr[hi]，必然lo和hi中间的元素都比hi的大
                    res = (res + (mid - lo + 1)) % 1000000007;
                    nums[idx++] = arr[hi++]; 
                }
            }
            // 继续收集结果
            while(lo <= mid) {
              nums[idx++] = arr[lo++]; 
            }
          while(hi <= right) {
                nums[idx++] = arr[hi++]; 
            }
            // 并到原来的数组中
            for(int i = 0; i <= right - left; i++) {
                arr[i + left] = nums[i];
            }
        }
    }
    ```

* [两个链表第一个公共节点](https://www.nowcoder.com/practice/6ab1d9a29e88450685099d45c9e31e46?tpId=13&tqId=11189&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 输入两个链表，找出它们的第一个公共结点。（注意因为传入数据是链表，所以错误测试数据的提示是用其他方式显示的，保证传入数据是正确的）

    ```
    public class Solution {
        public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
            ListNode p1 = pHead1, p2 = pHead2;
            while(p1 != p2) {
                // 当他们走的步数刚好相等时，两个指针第一次相遇
                p1 = (p1 == null ? pHead2 : p1.next); 
                p2 = (p2 == null ? pHead1 : p2.next); 
            }
            return p1;
        }
    }
    ```

* [数字在升序数组中出现的次数](https://www.nowcoder.com/practice/70610bf967994b22bb1c26f9ae901fa2?tpId=13&tqId=11190&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 统计一个数字在升序数组中出现的次数。

    ```
    public class Solution {
        public int GetNumberOfK(int [] array , int k) {
            // 通过浮点数，避免了重复元素导致前后两数的不一致性 
            // 寻找前一个数应该是最后<k的整数，而寻找后面一个则应该寻找第一个>k的整数
            return helper(array, k + 0.5) - helper(array, k - 0.5);
                    
        }
        
        // 找到num在arr的第一个下标
        int helper(int[] arr, double num) {
            // 二分法
            int lo = 0, hi = arr.length - 1;
            int mid = 0;
            while(lo <= hi) {
                mid = lo + ((hi - lo) >> 1);
                if(arr[mid] < num) {
                    lo = mid + 1;
                } else if(arr[mid] > num) {
                    hi = mid - 1;
                }
            }
            // 找到num在arr的第一个下标
            return lo; 
        }
        
    }
    ```

* [二叉树的深度](https://www.nowcoder.com/practice/435fb86331474282a3499955f0a41e8b?tpId=13&tqId=11191&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。

    ```
    public class Solution {
        public int TreeDepth(TreeNode root) {
            if(root == null) {
                return 0;
            }
            return Math.max(TreeDepth(root.left), TreeDepth(root.right)) + 1;
        }
    }
    ```

* [平衡二叉树](https://www.nowcoder.com/practice/8b3b95850edb4115918ecebdf1b4d222?tpId=13&tqId=11192&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tab=answerKey)

  * 输入一棵二叉树，判断该二叉树是否是平衡二叉树。

    在这里，我们只需要考虑其平衡性，不需要考虑其是不是排序二叉树

    **平衡二叉树**（Balanced Binary Tree），具有以下性质：它是一棵空树或它的左右两个子树的高度差的绝对值不超过1，并且左右两个子树都是一棵平衡二叉树。

    ```
    public class Solution {
        public boolean IsBalanced_Solution(TreeNode root) {
            if(root == null) {
                return true;
            }
            
            // -1表示有子树的高度差大于1
            return getDepth(root) != -1;
        }
        // 自底向上遍历
        int getDepth(TreeNode root) {
            if(root == null) {
                return 0;
            }
            // 先直接递归到最左叶子节点
            int left = getDepth(root.left);
            // 已经有子树高度差大于1，其他子树就不需要任何操作了
            if(left == -1) {
                return -1;
            }
            // 先直接递归到最右叶子节点
            int right = getDepth(root.right);
            if(right == -1) {
                return -1;
            }
            
            // 当递归到叶子节点后，假如左叶子节点比右叶子节点高一层，那么此时left=1, right=0。结果left-right也不会大于1
            // 若高2层，因为left比right多递归2次，因此2次+1将会比right高2个高度，返回-1
            // -1表示当前子树左右节点高度差大于1，后面的if判断表示剪枝
            return Math.abs(left - right) > 1 ? -1 : 1 + Math.max(left, right);
        }
    }
    ```

  



