---
layout: post
title:  "Algorithms Cheat Sheet"
date:   2021-01-01
categories: Algorithms
toc: true
---

{:toc}




| Data Type Ranges   |                                  |                   |
| ------------------ | :------------------------------- | ----------------- |
| int                | +- 2 billion (10 digits)         | 32 bits (4 bytes) |
| unsigned int       | 4 billion (10 digits)            |                   |
| short              | +- 32 thousand (5 digits)        | 16 bits (2 bytes) |
| unsigned short     | 65 thousand (5 digits)           |                   |
| long long          | +- 9 billion billion (19 digits) | 64 bits (8 bytes) |
| unsigned long long | 18 billion billion (19 digits)   |                   |
| char               | +- 127 (3 digits)                | 8 bits (1 byte)   |
| unsigned char      | 255 (3 digits)                   |                   |



## General useful functions:

Tie pair to int x, y variables:

```c++
std::tie(x, y) = ...pair...
auto [x, y] = ...pair...;
```

Template function

```c++
template typename<T> void selfMax(T &a, T b) {
    a = max(a, b); //<- provided max already implements the input templates
}
```

Min max element in a vector

```c++
const auto [min_, max_] = std::minmax_element(begin(vec), end(vec));
```

returns iterator, so use dereferenced *min_, *max_

```c++
auto last = std::unique(v.begin(), v.end()) 
//remove consecutive equal numbers and return the last unique position
```

To remove duplicates, sort vector, use unique, then erase from last to end `v.erase(last, v.end())`



```c++
/* Conversions*/ 

//int <-> char
char a = '5';
int ia = a - '0';
char a2 = ia + '0'; 
//convert UPPER CASE letter to lowercase and vice versa 
char UC = UC - 'A' + 'a'; 
char lc = lc - 'a' + 'A';
```



## Complexity (big O) overview

Table credit: [Bigocheat.com][bigocheatwebsite]

![bigocheat](/assets/bigocheat.png)

[bigocheatwebsite]: https://www.bigocheatsheet.com/

## Mathematics

#### Combinatorics

Factorial n! = 1*2*3*..*n (optional parameter, start *start+1*...*n) 

```c++
unsigned long long int factorial(int n, int start = 1){
    unsigned long long int fac = 1;
    for(int i=start+1; i<=n; i++)
        fac *= i;
    return fac;
}
```

Binomial coefficients (N choose k)   n k   = n!(n-k)!k! for 0kn; 0 otherwise 0! = 1;

```c++
int nCk(int n, int k){
    if(k<0 || k>n) return 0;
    int numerator = factorial(n, n-k);
    int denominator = factorial(k);
    return numerator/denominator ;
}
```

#### Logarithm rules

b^y = x -> log_b(x) = y

Product rule : 
log_b(x ∙ y) = log_b(x) + log_b(y) 

Quotient rule : 
log_b(x / y) = log_b(x) - log_b(y) 

Power rule : 
log_b(x^y) = y ∙ log_b(x) 

Switch base rule : 
log_b(c) = 1 / log_c(b) 

Change base rule : 
log_b(x) = log_c(x) / log_c(b)

#### Bit manipulation

Check if a number is even

```c++
//n % 2 == 0 <-- most commonly used
//n ^ 1 == 1 <-- number is even, XOR 1 is set, because number ends with 0
//n & 1 == 0 <-- number is even, & is not set, because number ends with 0

getBit (int num, int i) return (num & (1 << i) != 0)
//create a mask with zeros and 1 in ith bit (bit shift 1 by i) and and it to check if it's not null  

setBit (int num, int i) return (num | (1 << i) 
//create a mask with zeros and 1 in ith bit (bit shift 1 by i) and OR it to set it to 1 regardless of what it was

clearBit (int num, int i) return (num & ~(1 << i) ) 
//create a mask with ones and 0 in ith bit (bit shift 1 by i then negate) and it to set ith bit to 0. 

clearAllBitsFromItoMSB (int num, int i) return num & ((1 << i)-1 ) 
//create a mask with zeros and 1 in ith bit subtract 1, so that it's something like 00111 and And it

clearAllBitsFromIto0(int num, int i) return num & ~((1 << i+1)-1 ) 
//create a mask with zeros and 1 in ith bit subtract 1, so that it's something like 00111 and negate it then And it
```

Example XOR question: 
Given an array of integers, every element appears twice except for one. Find that single one. 
Solution: XOR will return 1 only on two different bits. So if two numbers are the same, XOR will return  0.So, all even occurrences will cancel out using XOR and the odd one remains. 

|                         | From cracking the code interview                             |
| ----------------------- | ------------------------------------------------------------ |
| x & (x-1)               | will clear the lowest set bit of x                           |
| x & ~(x-1)              | extracts the lowest set bit of x (all others are clear)      |
| x & (x + (1 << n)) = x  | with the run of set bits (possibly length 0) starting at bit n cleared. |
| x & ~(x + (1 << n)) =   | the run of set bits (possibly length 0) in x, starting at bit n. |
| x \| (x + 1) = x        | with the lowest cleared bit set.                             |
| x \| ~(x + 1) =         | extracts the lowest cleared bit of x (all others are set).   |
| x \| (x - (1 << n)) = x | with the run of cleared bits (possibly length 0) starting at bit n set. |
| x \| ~(x - (1 << n)) =  | the lowest run of cleared bits (possibly length 0) in x, starting at bit n are the only clear bits. |
|   

from (florian.github.io/xor-trick):
In-place swapping:
```c++
 x ^= y
 y ^= x
 x ^= y
```
Find missing number:
```c++
    missing = 0;
    //xor all numbers in range
    for(int i=1; i<n; i++)
        missing ^= i;
        
    //xor with all numbers in input array
    for(int v : input)
        missing ^= v;
    return missing;
```



## Trees

**Traversal**

- In order: left, root, right
- Pre order: root, left, right
- Post order: left, right, root

Two different trees may have the same in-order, pre-order traversal or post-order traversal. A combination is needed to describe a tree uniquely!   

Constructing tree from given traversal orders:

- Post order: used to define the root (last element)
- Pre order: used to define the root (first element) In order: when root is known, gives us the left subtree and right subtree (root is the splitting point)

**Array representation**

- Tree index shortcuts: 
  - i=0 --> root // 0 indexed 
  - parent(i) = (i-1)/2
  - leftChild(i) = 2i + 1
  - rightChild(i) = 2i + 2



**BST**

Goal: Fast insertion in a sorted data structure.

Observations: 

- Maximum **number of nodes at** the lowest **level** **h = (log N)** is number of nodes N/2; For every level above it divide by 2 (n/4, n/8 ... 1 root node);
- Operations/Queries:
  -  Insert: O(h) h: height of tree,  **log n** if it is a balanced tree!!!
  - Delete: O(h)
    - 1. case no child: delete pointer from parent
    - 2. case one child: like linked list, parent point to child. 
    - 3. case two children: replace with **min node in right** subtree OR **max node in left** subtree. Repeat until reaching case 1 or 2. 
  - Find/Search: O(h)
  -  Find min: O(h) - most left leaf
  -  Find max: O(h) - most right leaf
  -  Find next larger (successor): O(h) if node has a right tree: the successor is most left node of its right subtree. If a node doesn't have a right tree: the successor is the nearest ancestor where the given node is in its left subtree.
  -  Find previous (predecessor) : O(h) if node has a left tree: the predecessor is the max of the left subtree. if a node doesn't have a left tree: the predecessor is the nearest ancestor where the given node is in its right subtree. 

**The Depth / Height:**

The depth of a tree can be calculated *recursively* as max(depth(rightSubtree), depth(leftSubtree)) + 1. 

**Balanced BST**

1. AVL Tree: Height of left & right children differ by at most 1. Simple BST insert, then fix balance using rotations.
2. Red-Black-Tree
3. B-tree



**Right Rotation:(left becomes root)**
Let P be Q's left child.
Set Q's left child to be P's right child. 
Set P's right child to be Q. 

**Left Rotation: (right becomes root)**
Let Q be P's right child.
Set P's right child to be Q's left child.
Set Q's left child to be P.



**Delete a node from a BST**        

```c++
TreeNode* findMinimum(TreeNode* root) {    
    TreeNode* current = root;     
    while(current->left) {       
        current = current->left;        
    }     
    return current;  
}     
TreeNode* deleteNode(TreeNode* root, int key){    
    if(root == NULL)      
        return NULL;
    
    if(root->val > key)      
        root->left = deleteNode(root->left, key); 
    
    else if(root->val < key)      
        root->right = deleteNode(root->right, key);  

    //root->val == key
    else {      
        if(root->left && root->right){        
            TreeNode* toDelete = findMinimum(root->right); 
            //or find maximum in left subtree         
            root->val = toDelete->val;        
            root->right = deleteNode(root->right, root->val);      
        }      
        else if(root->left || root->right){        
            TreeNode* toDelete = root;        
            if(root->left)          
                root = root->left;        
            else          
                root = root->right;        
            delete toDelete;      
        }      
        else {        
            delete root;        
            root = NULL;      
        }    
    }    
    return root;  
}
```
<u>C++ Container</u>

std::map, std::set

**Count nodes in a subtree**

```c++
void dfs(int s, int e) {
    count[s] = 1;
    for (auto u : adj[s]) {
        if (u == e) 
            continue;
        dfs(u, s);
        count[s] += count[u];
    }
}
```

**Diameter** 

Diameter of a tree is the maximum length of a path between two nodes

```c++
int diameter(Node* root){  
    return (root == NULL) ? 0 : depth(root->left)+depth(root->right)+1;
}
```

Get the maximum path (counting edges) between two nodes (diameter) of a tree, not passing through the root necessarily.

```c++
pair<int, int> depthAndDiameter(Node* root) {   
    if(root == NULL) 
        return {0,0};
    
    pair<int, int> left = depthAndDiameter(root->left);  
    pair<int, int> right = depthAndDiameter(root->right);
    
    int maxdiameter = max({left.second, right.second, left.first+right.first+1});  
    int maxdepth = max(left.first, right.first)+1;

    return {maxdepth , maxdiameter};
}
```


## Tries

Tries are a type of tree that stores characters or strings.
Uses: Finding words, validating words(typos)

```c++

class Trie {
    unordered_map<char, Trie*> children;
    boolean isCompleteWord;void addWord(string word) {
        Trie*current = this;
        for(int i = 0; i<word.size(); i++){
            char c = word[i];
            if(current->children.find(c) == current->children.end() ) {
                Trie*n = new Trie();
                current->children.insert({c, n});
            }
            current = current->children[c];
            if(i == word.size()-1)
                current->complete = true;
        }
    }
}
```




## Graphs

**Problem examples:**

- Count the islands in a 2D matrix Use DFS or BFS, visited matrix, count all connected ones once (by marking 1s as visited) 
  - Solution trick: replace the matrix on the first visit with some value instead of keeping a visited matrix.
- Go style capture surrounded regions x0x
  - Mark at the 4 borders any 'O' as 'B' (for border)
  - DFS or BFS neighboring 'O', mark them as 'B' (They are not surrounded by 'X' from all sides)
  - Iterate again over 2d matrix, replace 'B' with 'O' and any remaining 'O' with 'X'
- Find the shortest path, but with specific restrictions. Ex. 2 brothers driving, same person starts and finishes. Ex. blue and red paths, and cost to change path color is higher
  - Solution trick: Transform the problem to **3D** i.e. duplicate nodes for each state. Ex. graph with blue/red paths, even nodes/odd nodes etc.

**General hack:**

Add neighbors in 2d matrix:

- 4 neighbors: insert 4 neighbors (i,j+1)(i+1, j)(i,j-1)(i-1,j) to a vector and loop it
- 8 neighbors: nested loop i=-1 to i<2, j=-1 to j<2, skip i=j=0 



Shortest Paths Algorithms Overview

Credit [William Fiset][williamfisetyoutube]

![shortestpathsalg](/assets/shortestpathsalg.png)

[williamfisetyoutube]: https://www.youtube.com/watch?v=4NQ3HnhyNfQ



#### Topological Sort of a DAG (Kahn's algorithm)

- Add vertices V and their indegree to an array (vector<int> indegree(V))
- Keep a queue of vertices with zero indegree (queue<int> q)
- Remove the vertex from the graph and add it to the topological sorted array (vector<int> sorted);
- When a vertex is removed, remove one indegree from all it's dependent nodes, add them to the queue if indegree is zero.
- Continue until the queue is empty. If the graph has a cycle, the array of indegree will have some vertices with indegrees > 0

```c++
vector<int> kahnsort(vector<vector<int>> &graph) {
    vector<int> indegree(graph.size());
    for(vector<int> n : graph) {
        for(int v : n)
            indegree[v]++; 
    }
    vector<int> sorted;queue<int> q; 
    for(int i=0; i<indegree.size(); i++) {
        if(indegree[i] == 0) 
            q.push(i); 
    }

    while(!q.empty()){
        int v = q.front(); q.pop();
        sorted.push_back(v);

        for(int n : graph[v]){
            indegree[n]--;
            if(indegree[n] <= 0)
                q.push(n); 
        }
    }
    if(sorted.size() != graph.size()) // cycle/no topological order
        return vector<int>();
    
    return sorted;
}
```


#### Minimum Spanning Trees (using DSU)

Pruning edges from a graph, by creating a tree with all vertices.
DSU : Disjoint-Set-Union

```c++
unordered_map<int, int> parents;
unordered_map<int, int> size; //union by size
unordered_map<int, int> ranks; //union by rank (attach the shorter tree to the root of the taller tree)

void make_set(const vector<int> &vertics) {
    for(int i : vertices) {
        parents[i]=i;
        size[i]=1;   
    }
}
int find_set(int n) {
     if(parents[n]==n) 
         return n;
     parents[n]=find_set(parents[n]); //path compression
     return parents[n];
}
void dsu_union(int a, int b) {
    a=find_set(a);
    b=find_set(b);//union by size 
    
    if(size[b] > size[a]) 
        swap(a,b);

    parents[b]=a;
    size[a] += size[b];
}
```


#### Kruskal's algorithm (used for sparse graphs)

```pseudocode
algorithm Kruskal(G)
    A := ∅
    for each v ∈ G.V do
        MAKE-SET(v)
    for each (u, v) in G.E ordered by weight(u, v), increasing do
        if FIND-SET(u) ≠ FIND-SET(v) then
           A := A ∪ {(u, v)}
           UNION(FIND-SET(u), FIND-SET(v))
    return A
```


Complexity: O(E log V) 


Prim's algorithm (used for dense (many edges) graphs) 

```c++
priority_queue<pair<int,int>, vector<pair<int,int>>, greater<pair<int,int>>> q;//minheap edge, adjacency list

unordered_set<int> visited;

q.push(make_pair(0,1));

while(!q.empty()) {
    auto curr = q.top();
    q.pop();
    if(visited.find(curr.second)!=visited.end())
        continue;
    visited.insert(curr.second);
    
    totalCost += curr.first;
    for(auto& adj: graph[curr.second]){
        if(visited.find(adj.second)!=visited.end()) continue;
            q.push(adj);
    }
}
```


#### DFS Depth first search graph traversal

Space: O(V) - Time: O(V+E)
Main dfs function that calls every vertex in the set (in case of an unconnected vertex)

```c++
void dfs(vector<vector<int>> &adj, vector<int>VerticesSet){
    //parent aka visited vector. can be a map to know the parent pointers for
    //topological sort!
    vector<bool> parent(VerticesSet.size(), false); 
    //optional finished flag vector used to detect cycles
    vector<bool> finished(VerticesSet.size(), false); 
    //optional start time vector used to detect cross edges
    vector<int> startTime(VerticesSet.size()); //TODO
    //loop to visit all vertices
    for(int s : VerticesSet){
       if(!parent[s]){
        parent[s]=true; 
           dfs_visit(adj, s, parent, finished); 
        }
    }
}
```

Recursive call. Can be made to return a bool value if we are checking //cycles. Iterative implementation would use a stack!

```c++
 bool dfs_visit(vector<vector<int>> &adj, int s, vector<bool>&parent, vector<bool>&finished){
    //startTime[s] = ? TODO
    for(int v : adj[s]){ 
        //Node is a child that is not visited = tree edge
        if(parent[v] == false) {
            //Set parent to true, or s if it's a map
            parent[v] = true;
            //Recursive call to visit v
            if(!dfs_visit(adj, v, parent, finished))
               return false for cycles;
        }
     //Node is visited and finished = forward edge or cross edge  
    //An edge(s,v) is a cross edge, if startTime[s]>startTime[v]. 
    else if(finished[v])
        /*optional do something*/
    //Node is visited but not finished = backward edge (cycle)
    else if(!finished[v] && /* parent[v] != vertex for undirected graphs */)
        /*optional do something*/
    }
    finished[s] = true;
    return true;
}
```


#### BFS Breadth first search graph traversal 

Space: O(V) - Time: O(V+E)

Shortest Path BFS. 
//s is starting node. t is destination node

```c++
void BFS(map<int, vector<int>> adj, int s, int t){
    queue<int> frontier; //Queue to add children of the visited node
    frontier.push(s);
    map<int, int> level;//Optional to keep track of level
    level.insert(s, 0);
    map<int, int> parent; // aka visited, map is used to extract shorted path
    parent.insert(s, -1);while(!frontier.empty()) {
        int u = frontier.front();
        frontier.pop(); //remove from queue
        /*do something with u*/
        int i=level[u];//level of node
        //Loop adjacent vertices
        for(int v : adj[u]) {
            //If adjacent nodes are not visited (not in parent)
            if(parent.count(v)) {
                if(v==t)
                    /*destination reached*/
                parent.insert({v, u}); //assign u as parent
                level.insert({v, i+1}); //set their level
                frontier.push(v); //add to queue
            }
        }
    }
}
```
#### Level order traversal for trees 

```c++
void levelorder_traversal(TreeNode* root){ 
    int current_max_level = 0;
    queue<TreeNode*> q;
    q.push(root);
    int level = 0; 
    while(!q.empty()) { 
        level++;
        //loop nodes in this level
        for (int i = q.size(); i > 0; --i) {
            TreeNode* n = q.front(); q.pop(); 
            /*do something*/
            //add children to be processed in next level
            if (n->left)  q.push(n->left);
            if (n->right) q.push(n->right);
        } 
    }
}
```



## Heaps 

Lookup: O(1) - Pop operation: O(logN) Build:O(N) - Heapsort: O(NlogN)- Space: O(N)

A heap is an array structure visualized as a binary tree. The key of a node is >= than its children (maxheap) or <= than its children (minheap).

<u>C++ Container:</u>
MaxHeap(int) --> std::priority_queue<int>;
MinHeap(int) --> std::priority_queue<int, std::vector<int>, greater<int>>;

```c++
//custom sort 
sort(mMyClassVector.begin(), mMyClassVector.end(), 
     [](const MyClass & a, const MyClass & b) -> bool { 
         return a.mPropertyb.mProperty; 
     })

//Example: Custom Heap using lambda to compare elements.
auto myComparator = [](int l, int r) { return (l^2) < (r^2); }; 
priority_queue <int, vector<int>, decltype(myComparator)> myHeap;

//Example: Custom compare struct for custom types
struct Compare {
       bool operator()(CustomT* const & p1, CustomT* const & p2) {
           // return "true" if "p1" is ordered before "p2", e.g.
           return p1->val > p2->val;
       }
};
priority_queue <CustomT, vector<CustomT>, decltype(Compare)> myHeap(Compare);

//Example: without struct
auto compare = [](ListNode* a, ListNode*b) {
       return a->val > b->val;
};
priority_queue<ListNode*, vector<ListNode*>, decltype(compare)> heap(compare);
```



*Complexity:*
Building the heap: O(N)
Accessing the top element: O(1) constant
Pop() operation: O(logN)

Popping the top element causes the heap to fix itself(heapify).
The root node is replaced by the last node in the array, and the heap property is fixed recursively. Time=O(logN)



## Linked lists (double and single) 

Search/Access: O(N) - Delete: O(1) - Insert:O(1) - Space: O(N)
<u>C++ Container:</u>
Single Linked List: std::forward_list<T>
Double Linked List: std::list<T>

Why is reversing useful? 

- To loop through a single list backwards!
- To find the topological sort of a DFS path if a linked list was used instead of a stack.

Recursive:
```c++
ListNode* reverseASingleList(ListNode*A, ListNode *&head) {
    if(A == NULL)
        return NULL;
    if(A->next == NULL) {
         head = A;
         return A;
    }    
    ListNode* newNext = reverseList(A->next);
    newNext->next = A;
    A->next = NULL;
    return A;
}
```


Iterative:

```c++
    
ListNode *reverseList(ListNode *head) {
    if (head == NULL) 
        return NULL; 
    ListNode *currentNode = head;
    ListNode *previousNode = NULL;
    ListNode *nextNode; 
    while (currentNode != NULL) {
        nextNode = currentNode->next;
        currentNode->next = previousNode;
        previousNode = currentNode;
        currentNode = nextNode;
    }
    //loop ends when currentNode == NULL, therefore, reveresed   
    //head is previousNode. 
    return previousNode;
}
```


Fast and slow pointer (detect cycles, reach the end faster, ... )

## Backtracking (Combinations)

Algorithm's very close to DFS. Recursive, checking if the next path is valid. Backtracking is trying out all possibilities using recursion, exactly like bruteforce.
Pseudo Code example:

```pseudocode
boolean solve(Node n): 
   if n is a goal node
return true
   foreach option O possible from n 
      if solve(O) succeeds
          return true
   return false
```

Example recursion function(incomplete) :

```c++
void recurse(vector<T> &solution, T &vec, vector<T> &input, int index, const vector<T> &possibleNextChoices) {
    if(index == input.size()) //or some breaking/finishing condition
        solution.push_back(vec);
    else {
        T current = input[index];
        for(int i=0; i<possibleNextChoices.size(); i++) {
            vec.push_back(...);
            recurse(solution, input, A, i, possibleNextChoices);
            vec.pop_back();
        }
    }
}
```

### Tail Recursion

When the recursion is the last call in the function. Benefit: compiler optimization. 
Convert regular factorial function to tail recursive function:

```c++
int factorialRecursive(int n) {
    if(n == 0) 
        return 1;
    return n * factorialRecursive(n-1); //needs the return to compute the result 
}

int factorialTailRecursive(int n, int result) {
    if(n == 0) return result;
        return factorialTailRecursive(n-1, n*result);
}
```



## Dynamic Programming

What are good states? 

Would it help to solve backwards?

Solution approach/steps: 

- Define subproblems (suffixes, prefixes, substrings) --> # subproblem    
- Define the states (DP(i), DP(i,j), etc..
- Define initialization
- Define goal state
- Guess part of the solution     --> # guesses
- Recurrence    (solution for subproblem) --> (time per subproblem)
- Recurse & memoize or bottom up topological order (DAG)
  --> time = #subproblem * time per subproblem
- Solve original problem --> complete function (min/max formula)



#### DP by topic:

<u>Strings, sequences:</u>

- 2D matrix with the input as rows and cols headers
  - wild card matching    
  - edit distance /levenstein algorithm 
  - longest common subsequence

<u>Grids</u>

- Given a 2D grid, move from top left to bottom right (or cell x to cell y)
  Dungeon game
- Number of ways to reach target (grid includes obstacles, so not combinatoric)
  Minimum path sum from (0,0) to (x,y)

<u>Combinatorics in 1 D</u>

- Coin change and combinations
  - Given array of possible numbers/coins, count ways to sum up to target with or without permutation, or optimize minimum number to reach target
  - Coin change 1: number of ways to get to sum using given coins, order of coins matter (permutations of same coins count).
  - Coin change 2: number of ways to get to sum but without double counting permutations
    Tickets in a year: given travel dates, and different costs for ticket abos, find minimum cost for trip.

<u>Knapsack, yes/no</u> 

- Given an array of items as input, and a maximum capacity, optimize which items to pick.
- Unbounded Knapsack problem = coin change 2
- Suffix problems: Perfect information (Black jack)
  Given a deck of cards, stocks prices,... optimize solution.    
  Blackjack: maximize winnings given constraints (rules of the game for dealer, player)
- Buy-sell stocks: maximize winnings given certain constraints (buy and sell only once, several time, transaction fees, cooldown period)
- Text justification: given a list of words and maximum line size, split into good lines (optimize text justification). Given penalty for spaces per line. 
  Substrings problems
- Parenthesis for matrix multiplication to optimize multiplication.
- Woodcutting: optimal division of input pieces to maximize profit.

<u>Prefix problems:</u> 

- Longest increasing subsequence: find the longest non-sequential subsequence in given arra
- Shortest paths (Bellmann ford) 

<u>Other problems:</u>

- Longest palindrome sequence: find longest palindorme subsequence in string
- Coin Pick game: alternating players picking a coin from sequence, maximize collected coins
- Stairs: find ways to climb stairs to given level, taking 1 step or 2 steps at a time
- Crazy Eight game: find the longest sequence of cards that are "like" each other. "like"=same suit, same number, or 8
- Wood Chucking: cut a wood piece length N at given markings, while paying the minimum cost. Price is the multiple of length of both pieces after cutting L1*L2
- Longest increasing subsequence:
  *trick: using two pointers O(n^2). Compare longest subsequence between pointer i and all smaller indices 0..j and store max in additional array at i.
      int longest = INT_MIN;
      vector<int> dp(A.size(), 1);
      for(int i=0; i<A.size(); i++){
          for(int j=0; j<i; j++){
              if(A[j] < A[i])
                  dp[i] = max(dp[i], dp[j]+1);
          }
          longest = max(longest, dp[i]);
      }

## Strings

**KMP Algorithm (find substring index)**
<u>Knuth-Morris-Pratt</u>
If string B is substring of A, return starting index of substring

```c++
int strstr(string A, string B) {
    int prev_start=0; 
    int i=prev_start;
    int j=0;
    while(i<A.size() && j<B.size()) {
       start = tempstart;
        if(A[i]==B[j]) {
            i++;
            j++;
        }
        else {
            j=0;
            i=++prev_start; 
        }
        if(j==B.size()) 
            return prev_start;
    }
    return -1;
}
```

**RABIN-KARP Algorithm** 

Optimized string matching using rolling hashes

The idea is to match a pattern with a window in the string (of pattern length) by comparing their hash values. Then only compare the characters if the hash value is the same. To be faster than brute force comparison, the hash code of the window should be calculated efficiently (in constant time instead of pattern length linear time). 

Rolling hash simple example for hashing digits (0-9)
H = c1 * a^m-1 + c2 * a^m-2 + ... + cm * a^0
m = 3 (pattern length)
a = a constant, here we are using 10, but  it's typically chosen as a power of 2
c = code of the character - here we will use the value of the digit

Find the followiing pattern in the given string s: 
string pattern = "456", string s = "937456"

pattern_hash = H("456") =code("4") * 10^2 + code("5") * 10^1 + doce("6") * 10^0 = 456

H("937") = code("9") * 10^2 + code("3") * 10^1 + code("7") * 10^0 = 937
H("374") = (hash("937") -  code("9")*10^2) * 10 + 4 = (937-900) * 10^2  + 4 = 37 * 100 + 4 = 374
...

H("456") = hash("745")  - code("7") * 10^2 + 6 = 456


pattern hash equals the window hash, now we can compare the individual characters of the pattern and the window in linear time (of pattern length)

```pseudocode
function RabinKarp(string s[1..n], string pattern[1..m])
    hpattern := hash(pattern[1..m]);
    for i from 1 to n-m+1
        hs := hash(s[i..i+m-1])
        if hs = hpattern
            if s[i..i+m-1] = pattern[1..m]
                return i
    return not found
```
Time complexity in detail:
hashing pattern : O(m), where m = pattern length
loop : O(n-m+1) = O(n)  where n = length of string
window hashing : O(1) with rolling hashing, O(m) with a naive hashing
comparision (hs=hpattern): O(m), should happen on
Total time = O(n+m) with good hashing, O(nm) if the window hash matches every time 

## Sorting Algorithms

**Bucket sort**

```c++
// Function to sort arr[] of size n using bucket sort 
void bucketSort(float arr[], int n) {   
    /* 1) Create n empty buckets */  
    vector<vector<float>> b(n);  
    

    /* 2) Put array elements in different buckets */  
    for (int i=0; i<n; i++) {    
        int bi = n*arr[i]; 
        // Index in bucket    
        b[bi].push_back(arr[i]);   
    }    
    /* 3) Sort individual buckets */  
    for (int i=0; i<n; i++)    
        sort(b[i].begin(), b[i].end());    
        
    /* 4) Concatenate all buckets into arr[] */  
    int index = 0;   
    for (int i = 0; i < n; i++)     
        for (int j = 0; j < b[i].size(); j++)      
            arr[index++] = b[i][j]; 

} 
```



## TODO:

- Randomized algorithms & probability 
- Graphs:
  - Working with graphs(not just adjacency graph, but also different representations)
  - Paths(dijkstra, A*, bellman ford, Floyed-Warshall)
  - Flow networks
- Dynamic programming
- Binary search
    - *Tip: always use the formula mid = L + (R-L)/2 to avoid overflow
    - Non-Obvious questions:
      - Painter Partition problem: minimum time to finish painting. 
      - Binary search in rotated array
