---
title: "Easy"
author: "Rui Wang"
date: "7/8/2017"
output: html_document
---

547. Friend Circles

```python

class Solution(object):
    def findCircleNum(self, M):
        """
        :type M: List[List[int]]
        :rtype: int
        """
        # soltuion 2
        # DFS
        
        cnt, N = 0, len(M)
        vset = set()
        def dfs(n):
            for x in range(N):
                if M[n][x] and x not in vset:
                    vset.add(x)
                    dfs(x)
        for x in range(N):
            if x not in vset:
                cnt += 1
                dfs(x)
        return cnt
    
        # BFS
        # cnt, N = 0, len(M)
        # vset = set()
        # def bfs(n):
        #     q = [n]
        #     while q:
        #         n = q.pop(0)
        #         for x in range(N):
        #             if M[n][x] and x not in vset:
        #                 vset.add(x)
        #                 q.append(x)
        # for x in range(N):
        #     if x not in vset:
        #         cnt += 1
        #         bfs(x)
        # return cnt
    
#         # solution 1
#         # get the number of x which f(x) = x
#         N = len(M)
#         f = range(N)

#         def find(x):
#             while f[x] != x: x = f[x]
#             return x
#         # just need to check the right top half
#         for x in range(N):
#             for y in range(x + 1, N):
#                 if M[x][y]: f[find(x)] = find(y)
#         return sum(f[x] == x for x in range(N))
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

638. Shopping Offers

```python
class Solution(object):
    def shoppingOffers(self, price, special, needs):
        """
        :type price: List[int]
        :type special: List[List[int]]
        :type needs: List[int]
        :rtype: int
        """
        # smart solution
        # dp state function:
        # dp[needs] = min(dp[needs], dp[needs - special[:-1]] + special[-1])

        dp = dict()
        def solve(tup):
            if tup in dp: return dp[tup]
            # intialize the solution without specials
            dp[tup] = sum(t * p for t, p in zip(tup, price))
            for sp in special:
                ntup = tuple(t - s for t, s in zip(tup, sp))
                if min(ntup) < 0: continue
                dp[tup] = min(dp[tup], solve(ntup) + sp[-1])
            return dp[tup]
        return solve(tuple(needs))

```

494. Target Sum

```python
class Solution(object):
    def findTargetSumWays(self, nums, S):
        """
        :type nums: List[int]
        :type S: int
        :rtype: int
        """
        # solution 1
        # state function: dp[i + 1][k + nums[i] * sgn] += dp[i][k]
        dp = collections.Counter()
        dp[0] = 1
        for n in nums:
            ndp = collections.Counter()
            for sgn in (1, -1):
                for k in dp.keys():
                    ndp[k + n * sgn] += dp[k]
            dp = ndp
        return dp[S]
```

491. Increasing Subsequences

```python
class Solution(object):
    def findSubsequences(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        # solution 1 using dp
        
        # dp = set()
        # for n in nums:
        #     for y in list(dp):
        #         if n >= y[-1]:
        #             dp.add(y + (n,))
        #     dp.add((n,))
        # return list(e for e in dp if len(e) > 1)
    
        # solution 2 using backtrack
        self.res = []
        def helper(temp, nums, start):
            if len(temp)>1: self.res.append(temp[:])
            # add a set to avoid duplicates
            # for example: [2,3,3] 2 with first 3 or second 3 only counts once
            used = set()
            for i in range(start, len(nums)):
                if (not temp or temp[-1] <= nums[i]) and nums[i] not in used:
                    temp.append(nums[i])
                    used.add(nums[i])
                    helper(temp, nums, i+1)
                    temp.pop()
                
        helper([], nums, 0) 
        return self.res
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

473. Matchsticks to Square

```python
class Solution(object):
    def makesquare(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        # over the OJ, add a visited set to make it check faster
        
        if not nums or len(nums) < 4: return False
        sumlength = sum(nums)
        if sumlength % 4 != 0: return False
        # reverse the sort also make it faster if the test case is very biased like larger value is the side length
        nums.sort(reverse=True)
        
        def helper(nums, start, sums, target, visited):
            if start >= len(nums): return all(s == target for s in sums)
            state = tuple(sorted(sums)) # fix the state using sort
            if state in visited: return False
            for i in range(4):
                # backtrack
                if sums[i] + nums[start] > target: continue
                sums[i] += nums[start]
                if helper(nums, start + 1, sums, target, visited): return True
                sums[i] -= nums[start]
            visited.add(state)
            return False
        sums, visited = [0] * 4, set()
        res = helper(nums, 0, sums, sumlength/4, visited)
        return res
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

394. Decode String

```python
class Solution(object):
    def decodeString(self, s):
        """
        :type s: str
        :rtype: str
        """
        # very smart codes
        stack = []
        stack.append(["", 1])
        num = ""
        for ch in s:
            if ch.isdigit():
              num += ch
            elif ch == '[':
                # use a two element array
                stack.append(["", int(num)])
                num = ""
            elif ch == ']':
                st, k = stack.pop()
                # stack[-1][0] += st*k
                stack[-1][0] += ''.join([st]*k)
            else:
                stack[-1][0] += ch
        return stack[0][0]
```

337. House Robber III

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def rob(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # solution 1 overtime not pass 
        # if not root: return 0
        # sum = 0
        # if root.left:
        #     sum += self.rob(root.left.left) + self.rob(root.left.right)
        # if root.right:
        #     sum += self.rob(root.right.left) + self.rob(root.right.right)
        # return max(sum+root.val, self.rob(root.left) + self.rob(root.right))
        
        # change it to dict and store it
        
        return self.robsub(root, {})
    def robsub(self, root, map):
        if not root:
            return 0
        if root in map:
            return map[root]
        sum = 0
        if root.left:
            sum += self.robsub(root.left.left, map) + self.robsub(root.left.right, map)
        if root.right:
            sum += self.robsub(root.right.left, map) + self.robsub(root.right.right, map)
        sum = max(sum+root.val, self.robsub(root.left, map) + self.robsub(root.right, map))
        map[root] = sum
        
        return sum
    
        # solution 2 greedy dfs
#         def helper(root):
#             if not root:
#                 return [0,0]
            
#             res = [0,0]
#             left = helper(root.left)
#             right = helper(root.right)
#             res[0] = max(left[0], left[1]) + max(right[0], right[1])
#             res[1] = root.val + left[0] + right[0]
#             return res
            
#         res = helper(root)
#         return max(res[0], res[1])
```

332. Reconstruct Itinerary

```python
class Solution(object):
    def findItinerary(self, tickets):
        """
        :type tickets: List[List[str]]
        :rtype: List[str]
        """
        # # solution 1 using hash table and bfs
        # routes = collections.defaultdict(list)
        # for s, e in tickets:
        #     routes[s].append(e)
        # def solve(start):
        #     left, right = [], []
        #     for end in sorted(routes[start]):
        #         if end not in routes[start]:
        #             continue
        #         routes[start].remove(end)
        #         subroutes = solve(end)
        #         if start in subroutes:
        #             left += subroutes
        #         else:
        #             right += subroutes
        #     return [start] + left + right
        # return solve("JFK")
        
        # solution 2 
        targets = collections.defaultdict(list)
        for a, b in sorted(tickets)[::-1]:
            targets[a].append(b)
        route = []
        def visit(airport):
            while targets[airport]:
                visit(targets[airport].pop())
            route.append(airport)
        visit('JFK')
        return route[::-1]
```

576. Out of Boundary Paths

```python
class Solution(object):
    def findPaths(self, m, n, N, i, j):
        """
        :type m: int
        :type n: int
        :type N: int
        :type i: int
        :type j: int
        :rtype: int
        """
        # solution 1
        dp = [[[0]*n for x in range(m)] for k in range(N+1)]
        for k in range(1, N+1):
            for x in range(m):
                for y in range(n):
                    v1 = 1 if x == 0 else dp[k-1][x-1][y]
                    v2 = 1 if x == m-1 else dp[k-1][x+1][y]
                    v3 = 1 if y == 0 else dp[k-1][x][y-1]
                    v4 = 1 if y == n-1 else dp[k-1][x][y+1]
                    dp[k][x][y] = (v1+v2+v3+v4)%(10**9+7)
                    
        return dp[N][i][j]
    
    
        # solution 2
        # dp[t + 1][x + dx][y + dy] += dp[t][x][y]    其中t表示移动的次数，dx, dy 取值 (1,0), (-1,0), (0,1), (0,-1)
        # MOD = 10**9 + 7
        # dz = zip((1, 0, -1, 0), (0, 1, 0, -1))
        # dp = [[0] *n for x in range(m)]
        # dp[i][j] = 1
        # ans = 0
        # for t in range(N):
        #     ndp = [[0] *n for x in range(m)]
        #     for x in range(m):
        #         for y in range(n):
        #             for dx, dy in dz:
        #                 nx, ny = x + dx, y + dy
        #                 if 0 <= nx < m and 0 <= ny < n:
        #                     ndp[nx][ny] = (ndp[nx][ny] + dp[x][y]) % MOD
        #                 else:
        #                     ans = (ans + dp[x][y]) % MOD
        #     dp = ndp
        # return ans
```

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

105. Construct Binary Tree from Preorder and Inorder Traversal

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        # preorder: root left right
        # inorder: left root right
        if not inorder or not preorder:
            return None
        
        root = TreeNode(preorder.pop(0))
        ind = inorder.index(root.val)
        # forward build tree
        # left right
        root.left = self.buildTree(preorder, inorder[:ind])
        root.right = self.buildTree(preorder, inorder[ind+1:])

        return root
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

129. Sum Root to Leaf Numbers

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sumNumbers(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # solution 1
        # dfs + stack
        if not root:
            return 0
        stack, res = [(root, root.val)], 0
        while stack:
            node, value = stack.pop()
            if node:
                if not node.left and not node.right:
                    res += value
                if node.right:
                    stack.append((node.right, value*10+node.right.val))
                if node.left:
                    stack.append((node.left, value*10+node.left.val))
        return res
        
        # solution 2
        # bfs + queue
        # if not root:
        #     return 0
        # queue, res = collections.deque([(root, root.val)]), 0
        # while queue:
        #     node, value = queue.pop(0)
        #     if node:
        #         if not node.left and not node.right:
        #             res += value
        #         if node.left:
        #             queue.append((node.left, value*10+node.left.val))
        #         if node.right:
        #             queue.append((node.right, value*10+node.right.val))
        # return res
    
        # solution 3
        # recursively 
#         self.res = 0

#         def dfs(root, value):
#             if root:
#                 if not root.left and not root.right:
#                     self.res += value*10 + root.val
#                 dfs(root.left, value*10+root.val)
#                 dfs(root.right, value*10+root.val)

#         dfs(root, 0)
#         return self.res
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

117. Populating Next Right Pointers in Each Node II

```python
# Definition for binary tree with next pointer.
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None

class Solution:
    # @param root, a tree link node
    # @return nothing
    def connect(self, root):
        while root:
            dummy = TreeLinkNode(0)
            cur = dummy
            while root:
                if root.left:
                    cur.next = root.left
                    cur = cur.next
                if root.right:
                    cur.next = root.right
                    cur = cur.next
                root = root.next
            # reach the next level
            root = dummy.next
```

116. Populating Next Right Pointers in Each Node

```python
# Definition for binary tree with next pointer.
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None

class Solution:
    # @param root, a tree link node
    # @return nothing
    def connect(self, root):
        if not root:
            return
        pre = root
        cur = None
        while pre.left:
            cur = pre
            while cur:
                cur.left.next = cur.right
                if cur.next:
                    cur.right.next = cur.next.left
                cur = cur.next
            pre = pre.left
```

114. Flatten Binary Tree to Linked List

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        """
        # solution 1
        # top - down
        # if not root: return
        # cur = root
        # while cur:
        #     if cur.left:
        #         pre = cur.left
        #         while pre.right:
        #             pre = pre.right
        #         pre.right = cur.right
        #         cur.right = cur.left
        #         cur.left = None
                
        #     cur = cur.right
        
        # solution 2
        # dfs bottom - up
        self.pre = None
        def dfs(root):
            if not root: return
            dfs(root.right)
            dfs(root.left)
            
            root.right = self.pre
            root.left = None
            self.pre = root
        dfs(root)
```

113. Path Sum II

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: List[List[int]]
        """
        self.res = []

        if not root:
            return self.res
        
        
        def dfs(root, sum, stack):
            stack.append(root.val)
            if not root.left and not root.right and sum == root.val:
                self.res.append(stack[:])
            if root.left:
                dfs(root.left, sum-root.val, stack)
            if root.right:
                dfs(root.right, sum-root.val, stack)
            stack.pop()
        
        dfs(root, sum, [])
        
        return self.res
```

106. Construct Binary Tree from Inorder and Postorder Traversal

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        # postorder: left right root
        # inorder left root right
        if not inorder or not postorder:
            return None
        root = TreeNode(postorder.pop())
        ind = inorder.index(root.val)
        root.right = self.buildTree(inorder[ind+1:], postorder)
        root.left = self.buildTree(inorder[:ind], postorder)
        return root
```

109. Convert Sorted List to Binary Search Tree

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sortedListToBST(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """
        if not head:
            return None
        return self.tobst(head, None)

    def tobst(self, head, tail):
        if head == tail: return None
    
        fast = slow = head
        while fast != tail and fast.next != tail:
            slow = slow.next
            fast = fast.next.next

        node = TreeNode(slow.val)
        node.left = self.tobst(head, slow)
        node.right = self.tobst(slow.next, tail)

        return node
```

98. Validate Binary Search Tree

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True
        stack = []
        pre = None
        
        while root or stack:
            # go the smallest node
            while root:
                stack.append(root)
                root = root.left
            
            root = stack.pop()
            if pre and pre.val >= root.val:
                return False
            pre = root
            root = root.right
        return True
```
