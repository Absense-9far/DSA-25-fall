#### 不定行输入：
```Python
import sys
for line in sys.stdin
``````

#### 二分查找 Binary Search

常见于求“合理安排使得最大/小值最小/大”等
①读题，确定变量有哪些，尤其是划分变量（要被不断二分的变量，一般就是那个要有最小值的最大值，要有最大值的最小值）
②给出一个初始上界和初始下界
③确定check函数，给出当前划分变量是否过大或过小（需要具体设计实现）
④循环，输出答案；注意边界
```Python
while low <= high:  
    mid = (low + high) // 2  
    if not check(mid): 不满足条件的先动 
        low = mid + 1  
    else:  
        high = mid - 1
print(low) return的是先动部分
```

Three Laws：
1. 递归算法必须有一个**基准情形(如f(0))**；
2. 递归算法必须改变自变量状态，使它朝着基准情形前进；
3. 递归算法必须重复调用自身。

e.g.进制转换：
```Python
def to_str(n, base):
    # 定义用于转换的字符序列
    convert_string = "0123456789ABCDEF"
    # 基准情形：如果 n 小于基数，则直接返回对应的字符
    if n < base:
        return convert_string[n]
    else:
        # 递归调用：先处理商，再处理余数
        # 通过延迟连接操作，确保结果的顺序是正确的
        return to_str(n // base, base) + convert_string[n % base]

# 示例
print(to_str(10, 2))  # 输出: "1010"
print(to_str(255, 16))  # 输出: "FF"
```

递归是实现DFS的一种常用方式。
##### Lake Counting：
```
10 12
W........WW.
.WWW.....WWW
....WW...WW.
.........WW.
.........W..
..W......W..
.W.W.....WW.
W.W.W.....W.
.W.W......W.
..W.......W.
```
（九宫格内算相连，数总结构数）
```
3
```
#### DFS
```python
import sys
sys.setrecursionlimit(20000)
def dfs(x,y):
	#标记，避免再次访问
    field[x][y]='.'
    for k in range(8):
        nx,ny=x+dx[k],y+dy[k]
        #范围内且未访问的lake
        if 0<=nx<n and 0<=ny<m\
                and field[nx][ny]=='W':
            #继续搜索
            dfs(nx,ny)
n,m=map(int,input().split())
field=[list(input()) for _ in range(n)]
cnt=0
dx=[-1,-1,-1,0,0,1,1,1]
dy=[-1,0,1,-1,1,-1,0,1]
for i in range(n):
    for j in range(m):
        if field[i][j]=='W':
            dfs(i,j)
            cnt+=1
print(cnt)
```
##### 八皇后问题：
```Python
ans = []
def queen_dfs(A, cur=0):          #考虑放第cur行的皇后
    if cur == len(A):             #如果已经放了n个皇后，一组新的解产生了
        ans.append(''.join([str(x+1) for x in A])) #注意避免浅拷贝
        return 
    '''即将检验在cur行col列放置'''
    for col in range(len(A)):     #将当前皇后逐一放置在不同的列，每列对应一组解
        for row in range(cur):    #逐一判定，与前面的皇后是否冲突
            #因为预先确定所有皇后一定不在同一行，所以只需要检查是否同列，或者在同一斜线上
            if A[row] == col or abs(col - A[row]) == cur - row:
                break
        else:                     #若都不冲突
            A[cur] = col          #放置新皇后，在cur行，col列
            queen_dfs(A, cur+1)	  #对下一个皇后位置进行递归
            
queen_dfs([None]*8)   
for _ in range(int(input())):
    print(ans[int(input()) - 1])
```
<span style="background-color:#00ff00ca">回溯注意撤销（此处A是覆盖式撤销，cur是传入cur+1不影响cur） 直接去append当然一会儿就爆掉</span>
也可以避免使用同一变量，在递归中传入的是A+[col]总之不出问题就行。
##### 递归优化：
1.深度优化：扩大深度限制：
```Python
sys.setrecursionlimit(1 << 30)
将递归深度限制指定为2^30
```
2.缓存函数的返回值：
```Python
from functools import lru_cache

@lru_cache(maxsize=None)
def f(...)...
```
<font color=#FF0212>比字典还要快很多。</font>
#### 动态规划：
```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        @cache
        def dfs(i):
            if i < 0:
                return 0
            return max(dfs(i-1),dfs(i-2)+nums[i])
        return dfs(len(nums)-1)
```
dfs的值可能是获取金额（dfs(i)+num），可能是成功与否，也可能是所用硬币个数（dfs(i)+1)。它就是问题本身
```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        target+= sum(nums)
        if target<0 or target%2 :
            return 0
        target //=2
'''本题dfs结果是总方案数，所以边界输出0/1且return用加法而非max'''
        @cache
        def dfs(i,c):
            if i<0:
                return 1 if c==0 else 0'''此处目标和'''
            return dfs(i-1,c)+dfs(i-1,c-nums[i])
        return dfs(len(nums)-1,target)

最长递增子序列的长度
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        # @cache
        # def dfs(i):
        #     res = 0
        #     for j in range(i):
        #         if nums[j]<nums[i]:
        #             res = max(res,dfs(j))
        #     return res+1
        # return max(dfs(i) for i in range(len(nums)))
        f = [0]*len(nums)
        for i in range(len(nums)):
            for j in range(i):
                if nums[j]<nums[i]:
                    f[i]=max(f[i],f[j])
            f[i]+=1
        return max(f)
```

#### Stack：
```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack=[]
        dic = {')':'(',']':'[','}':'{'}
        for x in list(s):
            if x in dic.values():
                stack.append(x)
            if x in dic:
                if  (not stack) or stack[-1] != dic[x]:
                    return False
                else:
                    stack.pop()
        if stack:
            return False
        else:
            return True
```

### BFS
1. **初始化**：将起始节点加入队列，并将其标记为已访问。
2. **探索过程**：
    当队列非空时，重复以下操作：
    - 从队列中取出一个节点并访问它（例如，打印其值）。
    - 遍历该节点的所有未访问邻居：
        - 将每个未访问的邻居加入队列。
        - 将该邻居标记为已访问。
3. **终止条件**：重复第 2 步，直到队列为空。
```python
r,c,k = map(int,input().split())  
land = [input().strip() for i in range(r)]  
start,end = 0,0  
for i in range(r):  
    for j in range(c):  
        if land[i][j] == 'S':  
            start = (i,j)  
        elif land[i][j] == 'E':  
            end = (i,j)  
dx = [-1,1,0,0]  
dy = [0,0,1,-1]  
visited = [[[False]*(k+1) for i in range(c)] for j in range(r)]  
go = []  
visited[start[0]][start[1]][k] = True  
go.append((start[0], start[1], k, 0))  
def solve(num):  
    while num < len(go):  
        x,y,nowk,step= go[num]  
        num+=1  
        if (x,y) == end:  
            print(step)  
            return  
        for i in range(4):  
            nx = x+dx[i]  
            '''此处一定要新定义变量，不能沿用x+=，否则在遍历中会一直改变。  
            在每个for下面赋值nx、ny才是对的'''  
            ny = y+dy[i]  
            if 0<=nx<r and 0<=ny<c:  
                if land[nx][ny] == '.' or land[nx][ny] == 'E':  
                    if not visited[nx][ny][nowk]:  
                        visited[nx][ny][nowk] = True  
                        go.append((nx,ny,nowk,step+1))  
                elif land[nx][ny] == '#' and nowk > 0:  
                    if not visited[nx][ny][nowk-1]:  
                        visited[nx][ny][nowk-1] = True  
                        go.append((nx,ny,nowk-1,step+1))  
    print(-1)  
solve(0)
```

```Python
from collections import deque  
  
dx = [0,0,-1,1]  
dy = [1,-1,0,0]  
  
def bfs(graph,r,c,start,end):  
    visited = [[0]*c for _ in range(r)]  
    queue = deque()  
    queue.append((start[0],start[1],0))  
    visited[start[0]][start[1]] = 1  
    while queue:  
        x,y,step = queue.popleft()  
        if (x,y) == end:  
            return step  
        for i in range(4):  
            nx,ny = x+dx[i],y+dy[i]  
            if 0<=nx<r and 0<=ny<c and not visited[nx][ny] and graph[nx][ny]!='#':  
                visited[nx][ny] = 1  
                queue.append((nx,ny,step+1))
                #注意这里要是写成step+=1是改了四个for的step是错误的！  
    else:  
        return -1
关键是：同时处在队列里的是同一层，即steps相同所能到达的全部对象。广度完了再往下，因此第一个到达的是属于step-1的最快的一个，就是最短的一个。
```
- **最短路径查找**：（无权图）在遍历过程中记录每个节点的父亲，即可重构出最短路径。
```Python
father[start] = None
while queue:cur =pop;if cur == end:
        path = []
        while cur:path.append(cur),cur=father[cur]
        path.reverse()
```
无向图的环检测：
```python
from collections import deque
def has_cycle_bfs(graph):
    visited = set()
    for node in graph:
        if node not in visited:
            queue = deque([(node, -1)])
            visited.add(node)
            while queue:
                current, parent = queue.popleft()
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, current))
                    elif neighbor != parent:
                        return True
    return False
```
    
#### DFS
```PYthon
import sys
sys.setrecursionlimit(1000000)  # 设置为100万层
def dfs(v, visited):'''此处如果默认为set()会成为跟随函数的变量，取决于函数历史而非每次重建'''
    if visited is None:
        visited = set()
    visited.add(v)
    print(v, end=' ')
    for neighbour in neighbor[v]:
        if neighbour not in visited:
            dfs(neighbour, visited)
```
#### 拓扑排序
```python
    result = []
    queue = deque([node for node in graph if indegree[node] == 0])
    
    while queue:
        current = queue.popleft()
        result.append(current)
        
        for neighbor in graph[current]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    
    # 检查是否有环
    if len(result) != len(graph):
        return []  # 有环，无法拓扑排序
    return result
```

dijkstra贪心最短路径
```python
import heapq
class OptimizedDijkstra:
    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]
    def add_edge(self, u, v, w):
        self.adj[u].append((v, w))
    def shortest_path(self, start, end):
        # 距离数组
        dist = [float('inf')] * self.n
        dist[start] = 0
        
        # (距离, 节点)的优先队列
        pq = [(0, start)]
        
        while pq:
            current_dist, u = heapq.heappop(pq)
            
            # 如果当前距离大于记录的距离，跳过
            if current_dist > dist[u]:
                continue
            
            # 遍历邻居
            for v, weight in self.adj[u]:
                new_dist = current_dist + weight
                
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))
        
        return dist[end] if dist[end] != float('inf') else -1
```

#### 并查集
```python
def find(parent, i):
    if parent[i] != i:
        parent[i] = find(parent, parent[i])（顺手的事改一下，总之find结果一定对，个体可能小问题）
    return parent[i]

def union(parent, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)
    if xroot != yroot:
        parent[xroot] = yroot

n, m = map(int, input().split())
parent = list(range(n + 1))
for _ in range(m):
    i, j = map(int, input().split())
    union(parent, i, j)

count = sum(i == parent[i] for i in range(1, n + 1))
print(count)'''独立集个数'''
```



#### 最小生成树（并查集或prim）
（最小权的无环连通图）
```python
import sys

def prim_mst_adj_matrix(graph):
    """
    Prim算法实现（邻接矩阵版本）
    
    参数:
        graph: 邻接矩阵，graph[i][j]表示顶点i到j的权重，0表示无边
    
    返回:
        mst_edges: 最小生成树的边列表
        total_weight: 总权重
    """
    V = len(graph)
    
    # 初始化
    key = [sys.maxsize] * V  # 存储连接到生成树的最小权重
    parent = [-1] * V  # 存储生成树结构
    in_mst = [False] * V  # 标记顶点是否在生成树中
    
    # 从顶点0开始
    key[0] = 0
    
    mst_edges = []
    total_weight = 0
    
    for _ in range(V):
        # 找到不在MST中且key值最小的顶点
        min_key = sys.maxsize
        u = -1
        for v in range(V):
            if not in_mst[v] and key[v] < min_key:
                min_key = key[v]
                u = v
        
        # 将顶点u加入MST
        in_mst[u] = True
        total_weight += key[u]
        
        # 记录边（排除起始顶点）
        if parent[u] != -1:
            mst_edges.append((parent[u], u, graph[parent[u]][u]))
        
        # 更新相邻顶点的key值
        for v in range(V):
            if graph[u][v] > 0 and not in_mst[v] and graph[u][v] < key[v]:
                key[v] = graph[u][v]
                parent[v] = u
    
    return mst_edges, total_weight
```
#### 验证二分查找树
```python
def check(root,left,right):
    if root is None:
        return True
    x = roo.val
    return left<x<rihgt and check(root.left,left,x) and check(root.rihgt,x,right)
```
有序数组转二叉搜索树
```python
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        n = len(nums)
        if n == 0:
            return None
        left = self.sortedArrayToBST(nums[:n//2])
        right = self.sortedArrayToBST(nums[n//2+1:])
        return TreeNode(nums[n//2], left, right)
```
#### 树上DP
```python
def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        ans = 0
        def dfs(node: Optional[TreeNode]) -> int:
            if node is None:
                return -1  # 对于叶子来说，链长就是 -1+1=0
            l_len = dfs(node.left) + 1  # 左子树最大链长+1
            r_len = dfs(node.right) + 1  # 右子树最大链长+1
            nonlocal ans
            ans = max(ans, l_len + r_len)  # 两条链拼成路径
            return max(l_len, r_len)  # 当前子树最大链长
        dfs(root)
        return ans
'''重点：1是用单个叶子来举例 2这个dp计算的是以这个节点为根的最大子链长，ans要在中途想办法计算'''
```

#### 求高度和叶子数
```python
class TreeNode:
    def __init__(self):
        self.left = None
        self.right = None

def tree_height(node):
    if node is None:
        return -1  # 根据定义，空树高度为-1
    return max(tree_height(node.left), tree_height(node.right)) + 1

def count_leaves(node):
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return 1
    return count_leaves(node.left) + count_leaves(node.right)

n = int(input())  # 读取节点数量
nodes = [TreeNode() for _ in range(n)]
has_parent = [False] * n  # 用来标记节点是否有父节点

for i in range(n):
    left_index, right_index = map(int, input().split())
    if left_index != -1:
        nodes[i].left = nodes[left_index]
        has_parent[left_index] = True
    if right_index != -1:
        #print(right_index)
        nodes[i].right = nodes[right_index]
        has_parent[right_index] = True

# 寻找根节点，也就是没有父节点的节点
root_index = has_parent.index(False)
root = nodes[root_index]

# 计算高度和叶子节点数
height = tree_height(root)
leaves = count_leaves(root)

print(f"{height} {leaves}")
```
#### 按每个节点和自身子节点大小顺序遍历

```python
def traverse(x, tree):
    # 当前节点及其所有子节点值
    group = [x] + tree[x]
    # 从小到大排序
    for val in sorted(group):
        if val == x:
            print(x)
        else:
            traverse(val, tree)


n = int(input())
tree = {}
all_children = set()

for _ in range(n):
    line = list(map(int, input().split()))
    val = line[0]
    tree[val] = line[1:]
    all_children.update(line[1:])

# 根节点 = 未作为子节点出现的节点
root = (set(tree.keys()) - all_children).pop()

traverse(root, tree)
```