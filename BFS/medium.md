---
title: "Medium"
author: "Rui Wang"
date: "7/8/2017"
output: html_document
---

529. Minesweeper

```python
class Solution(object):
    def updateBoard(self, board, click):
        """
        :type board: List[List[str]]
        :type click: List[int]
        :rtype: List[List[str]]
        """
        # BFS using queue
        m = len(board)
        n = len(board[0])
        queue = [click]
        while queue:
            [row, col] = queue.pop(0)
            if board[row][col] == 'M':
                board[row][col] = 'X'
            else:
                count = 0
                for i in range(-1,2):
                    for j in range(-1,2):
                        if i == 0 and j == 0:
                            continue
                        r, c = row+i, col+j
                        if r < 0 or r >= m or c < 0 or c >= n:
                            continue
                        if board[r][c] == 'M' or board[r][c] == 'X':
                            count += 1
                if count > 0:
                    board[row][col] = str(count)
                else:
                    board[row][col] = 'B'
                    for i in range(-1,2):
                        for j in range(-1,2):
                            if i == 0 and j == 0:
                                continue
                            r, c = row+i, col+j
                            if r < 0 or r >= m or c < 0 or c >= n:
                                continue
                            if board[r][c] == 'E':
                                queue.append([r,c])
                                board[r][c] = 'B'
        return board
        
        # DFS using stack
        # m = len(board)
        # n = len(board[0])
        # stack = [click]
        # while stack:
        #     [row, col] = stack.pop()
        #     if board[row][col] == 'M':
        #         board[row][col] = 'X'
        #     else:
        #         count = 0
        #         for i in range(-1,2):
        #             for j in range(-1,2):
        #                 if i == 0 and j == 0:
        #                     continue
        #                 r, c = row+i, col+j
        #                 if r < 0 or r >= m or c < 0 or c >= n:
        #                     continue
        #                 if board[r][c] == 'M' or board[r][c] == 'X':
        #                     count += 1
        #         if count > 0:
        #             board[row][col] = str(count)
        #         else:
        #             board[row][col] = 'B'
        #             for i in range(-1,2):
        #                 for j in range(-1,2):
        #                     if i == 0 and j == 0:
        #                         continue
        #                     r, c = row+i, col+j
        #                     if r < 0 or r >= m or c < 0 or c >= n:
        #                         continue
        #                     if board[r][c] == 'E':
        #                         stack.append([r,c])
        #                         board[r][c] = 'B'
        # return board
```

103. Binary Tree Zigzag Level Order Traversal

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root: return []
        queue = [root]
        res = []
        reverse = False
        while queue:
            n =  len(queue)
            sublist = []
            for i in range(n):
                node = queue.pop()
                if node.left:
                    queue.insert(0, node.left)
                if node.right:
                    queue.insert(0, node.right)
                sublist.append(node.val)
            if not reverse:
                res.append(sublist)
            else:
                res.append(sublist[::-1])
            reverse = not reverse
        return res
```

279. Perfect Squares

```python
class Solution(object):
    # using dp
    _dp = [0]
    def numSquares(self, n):
        dp = self._dp
        while len(dp) <= n:
            dp += min(dp[-i*i] for i in range(1, int(len(dp)**0.5)+1)) + 1,
        return dp[n]
        
    # another solution is using number theory
    # def numSquares(self, n):
    #     """
    #     :type n: int
    #     :rtype: int
    #     """
    #     while n%4 == 0:
    #         n /= 4
    #     if n%8 == 7:
    #         return 4
    #     temp = self.square(n)
    #     for a in xrange(temp+1):
    #         b = self.square(n-a*a)
    #         if a*a + b*b == n:
    #             return int(a!=0) + int(b!=0)
    #     return 3

    # def square(self, n):
    #     r = n
    #     while r*r > n:
    #         r = (r+n/r)/2
    #     return r
```

417. Pacific Atlantic Water Flow

```python
class Solution(object):
    def pacificAtlantic(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """
        if not matrix:
            return []
        
        m = len(matrix)
        n = len(matrix[0])
        topEdge = [(0, y) for y in range(n)]
        leftEdge = [(x, 0) for x in range(m)]
        pacific = set(topEdge + leftEdge)
        bottomEdge = [(m - 1, y) for y in range(n)]
        rightEdge = [(x, n - 1) for x in range(m)]
        atlantic = set(bottomEdge + rightEdge)
        
        def bfs(vset):
            dz = zip((1, 0, -1, 0), (0, 1, 0, -1))
            queue = list(vset)
            while queue:
                hx, hy = queue.pop(0)
                for dx, dy in dz:
                    nx, ny = hx + dx, hy + dy
                    if 0 <= nx < m and 0 <= ny < n:
                        # it can flow from (hx, hy) to (nx, ny)
                        if matrix[nx][ny] >= matrix[hx][hy]:
                            if (nx, ny) not in vset:
                                queue.append((nx, ny))
                                vset.add((nx, ny))
        bfs(pacific)
        bfs(atlantic)
        result = pacific & atlantic
        return map(list, result)
```

515. Find Largest Value in Each Tree Row

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def largestValues(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        # it is the same problem with 102, 107
        # just add the max into res when treating the sublist
        # however, it take too many spaces
        # do not apply sublist, apply a temp max, make comparison everytime and attach the max value for each level

        res = []
        if not root: return []
        queue = [root]
        
        while queue:
            n = len(queue)
            # python only have sys.maxint for maximum of integer
            # we will use float('-inf') and float('inf')
            submax = float('-inf')
            for i in range(n):
                node  = queue.pop()
                submax = max(submax, node.val)
                if node.left:
                    queue.insert(0, node.left)
                if node.right:
                    queue.insert(0, node.right)
            res.append(submax)
        return res
        
        # solution 2
        # use the level, dfs
    #     self.res = []
    #     self.helper(root, 0)
    #     return self.res
    
    # def helper(self, root, level):
    #     if root:
    #         if level >= len(self.res):
    #             self.res.append(root.val)
    #         self.res[level] = max(self.res[level], root.val)
    #         self.helper(root.left, level + 1)
    #         self.helper(root.right, level + 1)
```

513. Find Bottom Left Tree Value

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def findBottomLeftValue(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        queue = [root]
        while queue:
            node = queue.pop()
            
            if node.right:
                queue.insert(0, node.right)
            if node.left:
                queue.insert(0, node.left)
                
        return node.val
```

310. Minimum Height Trees

```python
class Solution(object):
    def findMinHeightTrees(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        if n == 1: return [0] 
        adj = [set() for _ in xrange(n)]
        for i, j in edges:
            adj[i].add(j)
            adj[j].add(i)
    
        leaves = [i for i in xrange(n) if len(adj[i]) == 1]
    
        while n > 2:
            n -= len(leaves)
            newLeaves = []
            for i in leaves:
                j = adj[i].pop()
                adj[j].remove(i)
                if len(adj[j]) == 1: newLeaves.append(j)
            leaves = newLeaves
        return leaves
```

102. Binary Tree Level Order Traversal

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        # solution 1 DFS
        # attach the value at each level
        # if the height is larger than the list size, add [] for another level
    #     res = []
    #     self.levelhelper(res, root, 0)
    #     return res
        
    # def levelhelper(self, res, root, height):
    #     if root:
    #         if height >= len(res):
    #             res.append([])
    #         res[height].append(root.val)
    #         self.levelhelper(res, root.left, height + 1)
    #         self.levelhelper(res, root.right, height + 1)
    
        # solution 2 using queue
        res = []
        if not root: return res
        queue = [root]
        while queue:
            # how many node in the same level
            n = len(queue)
            sublevel = []
            # put all nodes from the next level in the queue
            for i in range(n):
                if queue[-1].left:
                    queue.insert(0, queue[-1].left)
                if queue[-1].right:
                    queue.insert(0, queue[-1].right)
                sublevel.append(queue.pop().val)
            res.append(sublevel)
        return res
```

210. Course Schedule II

```python
class Solution(object):
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        # BFS
        # dic = {i: set() for i in xrange(numCourses)}
        # neigh = collections.defaultdict(set)
        # for i, j in prerequisites:
        #     dic[i].add(j)
        #     neigh[j].add(i)
        # # queue stores the courses which have no prerequisites
        # queue = collections.deque([i for i in dic if not dic[i]])
        # count, res = 0, []
        # while queue:
        #     node = queue.popleft()
        #     res.append(node)
        #     count += 1
        #     for i in neigh[node]:
        #         dic[i].remove(node)
        #         if not dic[i]:
        #             queue.append(i)
        # return res if count == numCourses else []
        
        # another BFS
        degrees = [0] * numCourses
        childs = [[] for x in range(numCourses)]
        for pair in prerequisites:
            degrees[pair[0]] += 1
            childs[pair[1]].append(pair[0])
        courses = set(range(numCourses))
        flag = True
        ans = []
        while flag and len(courses):
            flag = False
            removeList = []
            for x in courses:
                if degrees[x] == 0:
                    for child in childs[x]:
                        degrees[child] -= 1
                    removeList.append(x)
                    flag = True
            for x in removeList:
                ans.append(x)
                courses.remove(x)
        return [[], ans][len(courses) == 0]
    
        # DFS
        # dic = collections.defaultdict(set)
        # neigh = collections.defaultdict(set)
        # for i, j in prerequisites:
        #     dic[i].add(j)
        #     neigh[j].add(i)
        # stack = [i for i in xrange(numCourses) if not dic[i]]
        # res = []
        # while stack:
        #     node = stack.pop()
        #     res.append(node)
        #     for i in neigh[node]:
        #         dic[i].remove(node)
        #         if not dic[i]:
        #             stack.append(i)
        #     dic.pop(node)
        # return res if not dic else []
```

207. Course Schedule

```python
class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        # BFS
        # dic = {i: set() for i in xrange(numCourses)}
        # neigh = collections.defaultdict(set)
        # for i, j in prerequisites:
        #     dic[i].add(j)
        #     neigh[j].add(i)
        # # queue stores the courses which have no prerequisites
        # queue = collections.deque([i for i in dic if not dic[i]])
        # count, res = 0, []
        # while queue:
        #     node = queue.popleft()
        #     res.append(node)
        #     count += 1
        #     for i in neigh[node]:
        #         dic[i].remove(node)
        #         if not dic[i]:
        #             queue.append(i)
        # return count == numCourses
        
        # another BFS
        degrees = [0] * numCourses
        childs = [[] for x in range(numCourses)]
        for pair in prerequisites:
            degrees[pair[0]] += 1
            childs[pair[1]].append(pair[0])
        courses = set(range(numCourses))
        flag = True
        while flag and len(courses):
            flag = False
            removeList = []
            for x in courses:
                if degrees[x] == 0:
                    for child in childs[x]:
                        degrees[child] -= 1
                    removeList.append(x)
                    flag = True
            for x in removeList:
                courses.remove(x)
        return len(courses) == 0
    
        # DFS
        # dic = collections.defaultdict(set)
        # neigh = collections.defaultdict(set)
        # for i, j in prerequisites:
        #     dic[i].add(j)
        #     neigh[j].add(i)
        # stack = [i for i in xrange(numCourses) if not dic[i]]
        # res = []
        # while stack:
        #     node = stack.pop()
        #     res.append(node)
        #     for i in neigh[node]:
        #         dic[i].remove(node)
        #         if not dic[i]:
        #             stack.append(i)
        #     dic.pop(node)
        # return True if not dic else False
```

200. Number of Islands

```python
class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        # bfs using queue
        ans = 0
        if not len(grid):
            return ans
        m, n = len(grid), len(grid[0])
        visited = [ [False] * n for x in range(m) ] # m * n
        for x in range(m):
            for y in range(n):
                if grid[x][y] == '1' and not visited[x][y]:
                    ans += 1
                    self.bfs(grid, visited, x, y, m, n)
        return ans

    def bfs(self, grid, visited, x, y, m, n):
        dz = zip( [1, 0, -1, 0], [0, 1, 0, -1] )
        queue = [ (x, y) ]
        visited[x][y] = True
        while queue:
            front = queue.pop(0)
            for p in dz:
                np = (front[0] + p[0], front[1] + p[1])
                if self.isValid(np, m, n) and grid[np[0]][np[1]] == '1' and not visited[np[0]][np[1]]:
                    visited[ np[0] ][ np[1] ] = True
                    queue.append(np)

    def isValid(self, np, m, n):
        return np[0] >= 0 and np[0] < m and np[1] >= 0 and np[1] < n
```
199. Binary Tree Right Side View

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        # solution 1 use queue
        # res = []
        # if not root: return res
        # queue = [root]
        # while queue:
        #     n = len(queue)
        #     res.append(queue[0].val)
        #     for i in range(n):
        #         node = queue.pop()
        #         if node.left:
        #             queue.insert(0, node.left)
        #         if node.right:
        #             queue.insert(0, node.right)
        # return res
        
        # solution 2 use dfs
        
        self.res = []
        def helper(root, level):
            if not root: return []
            if level == len(self.res):
                self.res.append(root.val)
            helper(root.right, level + 1)
            helper(root.left, level + 1)
        helper(root, 0)
        return self.res
```

133. Clone Graph

```python
# Definition for a undirected graph node
# class UndirectedGraphNode:
#     def __init__(self, x):
#         self.label = x
#         self.neighbors = []

class Solution:
    # @param node, a undirected graph node
    # @return a undirected graph node
    def cloneGraph(self, node):
        # dfs using stack
        if node is None:
            return None
        needle = UndirectedGraphNode(node.label)
        # use the dict {label : node}
        # neighbors to store all the connected nodes
        nodeDict = {node.label : needle}
        stack = [node]
        while stack:
            top = stack.pop()
            cnode = nodeDict[top.label]
            for n in top.neighbors:
                if n.label not in nodeDict:
                    nodeDict[n.label] = UndirectedGraphNode(n.label)
                    stack.append(n)
                cnode.neighbors.append(nodeDict[n.label])
        return needle
```

130. Surrounded Regions

```python
class Solution(object):
    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        # 从边上开始搜索，如果是'O'，那么搜索'O'周围的元素，并将'O'置换为'D'，这样每条边都DFS或者BFS一遍。而内部的'O'是不会改变的。这样下来，没有被围住的'O'全都被置换成了'D'，被围住的'O'还是'O'，没有改变。然后遍历一遍，将'O'置换为'X'，将'D'置换为'O'。
        
        # solution 1
        if not board:
            return

        n, m = len(board), len(board[0])
        q = [ij for k in range(max(n,m)) for ij in ((0, k), (n-1, k), (k, 0), (k, m-1))]
        while q:
            i, j = q.pop()
            if 0 <= i < n and 0 <= j < m and board[i][j] == 'O':
                board[i][j] = 'N'
                q += (i, j-1), (i, j+1), (i-1, j), (i+1, j)

        board[:] = [['XO'[c == 'N'] for c in row] for row in board]
        
        # solution 2
        # BFS
        
        # edge case
#         if not board:
#             return
        
#         def fill(x, y):
#             if x<0 or x>m-1 or y<0 or y>n-1 or board[x][y] != 'O': return
#             queue.append((x,y))
#             board[x][y]='D'
            
#         def bfs(x, y):
#             if board[x][y]=='O':
#                 queue.append((x,y))
#                 fill(x,y)
#             while queue:
#                 curr = queue.pop(0)
#                 i, j = curr[0], curr[1]
#                 fill(i+1,j)
#                 fill(i-1,j)
#                 fill(i,j+1)
#                 fill(i,j-1)
        
#         m=len(board); n=len(board[0]); queue=[]
#         for i in range(n):
#             bfs(0,i); bfs(m-1,i)
#         for j in range(1, m-1):
#             bfs(j,0); bfs(j,n-1)
#         for i in range(m):
#             for j in range(n):
#                 if board[i][j] == 'D': board[i][j] = 'O'
#                 elif board[i][j] == 'O': board[i][j] = 'X'
```

542. 01 Matrix

```python
class Solution(object):
    def updateMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """
        # solution 2
        if not matrix:
            return None
        m = len(matrix)
        n = len(matrix[0])
        dist = [[sys.maxint]*n for i in range(m)]
        
        queue = []

        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    dist[i][j] = 0
                    queue.append((i,j))
        dir = [[-1,0],[1,0],[0,-1],[0,1]]
        while queue:
            (x, y) = queue.pop(0)
            for i in range(4):
                newx = x + dir[i][0]
                newy = y + dir[i][1]
                if newx >= 0 and newx < m and newy >=0 and newy < n:
                    if dist[newx][newy] > dist[x][y] + 1:
                        dist[newx][newy] = dist[x][y] + 1
                        # put this indexer of smaller dist into queue
                        queue.append((newx, newy))
        return dist
                             
        # solution 1
        # scan twice
#         if not matrix:
#             return None
#         m = len(matrix)
#         n = len(matrix[0])
        
#         dist = [[sys.maxint]*n for i in range(m)]
        
#         for i in range(m):
#             for j in range(n):
#                 if matrix[i][j] == 0:
#                     dist[i][j] = 0
#                 # dist[i][j] = 1
#                 else:
#                     if i > 0:
#                         dist[i][j] = min(dist[i][j], dist[i-1][j]+1)
#                     if j > 0:
#                         dist[i][j] = min(dist[i][j], dist[i][j-1]+1)
#         for i in range(m)[::-1]:
#             for j in range(n)[::-1]:
#                 if matrix[i][j] == 0:
#                     dist[i][j] = 0
#                 # dist[i][j] = 1
#                 else:
#                     if i < m-1:
#                         dist[i][j] = min(dist[i][j], dist[i+1][j]+1)
#                     if j < n-1:
#                         dist[i][j] = min(dist[i][j], dist[i][j+1]+1)
#         return dist
```