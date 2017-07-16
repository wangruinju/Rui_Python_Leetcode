---
title: "Medium"
author: "Rui Wang"
date: "7/8/2017"
output: html_document
---

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

413. Arithmetic Slices

```python
class Solution(object):
    def numberOfArithmeticSlices(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        # my original solution, very bad but pass
        # cnt = []
        # ans = 0
        # for i in xrange(len(A)-2):
        #     if A[i] + A[i+2] == A[i+1]*2:
        #         cnt.append(1)
        #     else:
        #         cnt.append(0)
        # print cnt
        # while cnt:
        #     ans += sum(cnt)
        #     for i in xrange(len(cnt)-1):
        #         cnt[i] = cnt[i]&cnt[i+1]
        #     cnt.pop()
        # return ans
        
        # best solution put all in one loop
        cur = 0
        sum = 0
        for i in xrange(2, len(A)):
            if A[i] - A[i-1] == A[i-1] - A[i-2]:
            # 0+1+2+... if continous
                cur += 1
                sum += cur
            # discontinous then reset cur = 0
            else:
                cur = 0
        return sum
```

516. Longest Palindromic Subsequence

```python
class Solution(object):
    def longestPalindromeSubseq(self, s):
        """
        :type s: str
        :rtype: int
        """
        # solution 1
        # state function: dp[i][j]表示s[i .. j]的最大回文子串长度
        # dp[i][j] = dp[i + 1][j - 1] + 2           if s[i] == s[j]
        # dp[i][j] = max(dp[i][j - 1], dp[i + 1][j])    otherwise
        # size = len(s)
        # dp = [[0]*size for i in range(size)]
        # for i in range(size)[::-1]:
        #     dp[i][i] = 1
        #     for j in range(i+1, size):
        #         if s[i] == s[j]:
        #             dp[i][j] = dp[i+1][j-1]+2
        #         else:
        #             dp[i][j] = max(dp[i+1][j], dp[i][j-1])
        # return dp[0][-1]
    
    
        # solution 2
        
        # n = len(s)
        # res = 0
        # dp = [1]*n
        # for i in range(n)[::-1]:
        #     length = 0
        #     for j in range(i+1, n):
        #         temp = dp[j]
        #         if s[i] == s[j]:
        #             dp[j] = length+2
        #         length = max(length, temp)
        # for num in dp:
        #     res = max(res, num)
        # return res
        
        
        # solution 3 use cache
        n = len(s)
        self.dp = [[-1]*n for _ in range(n)]
        def helper(s,i,j):
            if self.dp[i][j] != -1: return self.dp[i][j]
            if i>j: return 0
            if i == j: return 1
            if s[i]==s[j]:
                self.dp[i][j] = helper(s, i+1, j-1)+2
            else:
                self.dp[i][j] = max(helper(s, i+1, j), helper(s, i, j-1))
            return self.dp[i][j]
        return helper(s, 0, n-1)
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

486. Predict the Winner

```python
class Solution(object):
    def PredictTheWinner(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        # smart codes
        # 函数solve(nums)计算当前玩家从nums中可以获得的最大收益，当收益>=0时，此玩家获胜
        # solve(nums) = max(nums[0] - solve(nums[1:]), nums[-1] - solve(nums[:-1]))
        cache = dict()
        def solve(nums):
            if len(nums) <= 1: return sum(nums)
            tnums = tuple(nums)
            if tnums in cache: return cache[tnums]
            cache[tnums] = max(nums[0] - solve(nums[1:]), nums[-1] - solve(nums[:-1]))
            return cache[tnums]
        return solve(nums) >= 0
```

91. Decode Ways

```python
class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        # solution 1 dp problem
        # from start to tail
        n = len(s)
        if n == 0:
            return 0
        res = [0]*(n+1)
        res[0] = 1
        res[1] = 1 if s[0] != '0' else 0
        for i in range(2, n+1):
            first = int(s[i-1:i])
            second = int(s[i-2:i])
            if first >= 1 and first <= 9:
                res[i] += res[i-1]
            if second >= 10 and second <= 26:
                res[i] += res[i-2]
        return res[-1]
        
        # solution 2
        # from tail to start
        
#         res[n] = 1
#         res[n-1] = 1 if s[n-1] != '0' else 0
        
#         for i in range(n-1)[::-1]:
#             if s[i] == '0': 
#                 continue
#             else:
#                 res[i] = res[i+1] + res[i+2] if int(s[i:i+2]) <= 26 else res[i+1]
#         return res[0]
```

95. Unique Binary Search Trees II

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def generateTrees(self, n):
        """
        :type n: int
        :rtype: List[TreeNode]
        """
        if n == 0: return []
        return self.dfs(1, n)
    
    def dfs(self, start, end):
        if start > end: return [None]
        res = []
        for rootval in range(start, end+1):
            lefttree = self.dfs(start, rootval-1)
            righttree = self.dfs(rootval+1, end)
            for i in lefttree:
                for j in righttree:
                    # preorder: root left right
                    root = TreeNode(rootval)
                    root.left = i
                    root.right = j
                    res.append(root)
        return res
```

96. Unique Binary Search Trees

```python
class Solution(object):
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        # for different root
        # if the root is i
        # then i, i+2 ... n will be on the right subtree, the count is hash[n-i]
        # the left subtree consists of 1, 2, i-1, the count is hash[i-1]
        # multiple for just root i
        hash = {0:1}
        index = 0
        while index <= n:
            sum = 0
            if index not in hash.keys():
                for i in range(index):
                    sum += hash[i]*hash[index-1-i]
                hash[index] = sum
            index += 1
        return hash[n]
```

474. Ones and Zeroes

```python
class Solution(object):
    def findMaxForm(self, strs, m, n):
        """
        :type strs: List[str]
        :type m: int
        :type n: int
        :rtype: int
        """
        dp = [[0] * (n + 1) for x in range(m + 1)]
        for s in strs:
            temp = collections.Counter(s)
            zero = temp['0']
            one = temp['1']
            for x in range(zero, m+1)[::-1]:
                for y in range(one, n+1)[::-1]:
                    dp[x][y] = max(dp[x - zero][y - one] + 1, dp[x][y])
        return dp[m][n]
```

467. Unique Substrings in Wraparound String

```python
class Solution(object):
    def findSubstringInWraproundString(self, p):
        """
        :type p: str
        :rtype: int
        """
        # solution 1
        count = [0]*26
        maxcur = 0
        for i in range(len(p)):
            if i>0 and ord(p[i]) - ord(p[i-1]) == 1 or ord(p[i-1]) - ord(p[i]) == 25:
                # if it is continuous
                maxcur += 1
            else:
                # if it is break
                maxcur = 1
            index = ord(p[i]) - ord('a')
            count[index] = max(count[index], maxcur)
        return sum(count)
        
        # solution 2
        # pattern = 'zabcdefghijklmnopqrstuvwxyz'
        # cmap = collections.defaultdict(int)
        # clen = 0
        # for c in range(len(p)):
        #     if c and p[c-1:c+1] not in pattern:
        #         clen = 1
        #     else:
        #         clen += 1
        #     cmap[p[c]] = max(clen, cmap[p[c]])
        # return sum(cmap.values())
```

120. Triangle

```python
class Solution(object):
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        n = len(triangle)
        minlen = triangle[-1]
        for layer in range(n-1)[::-1]:
            # for layer: it has layer+1 elements
            for i in range(layer+1):
                # only care minlen from index 0 to i
                # it will diminish to index 0 eventually
                # that's the minimum pathway
                minlen[i] = min(minlen[i], minlen[i+1]) + triangle[layer][i]
        return minlen[0]
```

64. Minimum Path Sum

```python
class Solution(object):
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        m = len(grid)
        n = len(grid[0])
        cur = [grid[0][0]]*m
        
        for i in xrange(1, m):
            cur[i] = cur[i-1] + grid[i][0]
        
        for j in xrange(1, n):
            cur[0] += grid[0][j]
            for i in xrange(1,m):
                cur[i] = min(cur[i-1], cur[i]) + grid[i][j]
        return cur[-1]
```

464. Can I Win

```python
class Solution(object):
    def canIWin(self, maxChoosableInteger, desiredTotal):
        """
        :type maxChoosableInteger: int
        :type desiredTotal: int
        :rtype: bool
        """
        if maxChoosableInteger >= desiredTotal: return True
        if maxChoosableInteger*(maxChoosableInteger + 1)/2 < desiredTotal: return False
        return self.helper(maxChoosableInteger, desiredTotal, 0, {})
    
    def helper(self, length, total, used, dict):
        if used in dict: return dict[used]
        # cur的第i位为1时，表示选择了数字i+1
        for i in range(length):
            cur = 1<<i
            # position i is not set, which means i is not used before
            if cur&used == 0:
                # smaller or the opponent can't win in the next step
                if total <= i+1 or not self.helper(length, total-(i+1), cur | used, dict):
                    dict[used] = True
                    return True
        dict[used] = False
        return False
```

139. Word Break

```python
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        res = [False]*(len(s)+1)
        res[0] = True
        # solution 1 check part of wordDict in s
        # for i in range(1, len(s)+1):
        #     for str in wordDict:
        #         if len(str) <= i:
        #             if res[i-len(str)] and s[i-len(str):i] == str:
        #                 res[i] = True
        #                 break
        
        # solution 2 check part of string in wordDict
        for i in range(1, len(s)+1):
            for j in range(i):
                if res[j] and s[j:i] in wordDict:
                    res[i] = True
                    break
        
        return res[-1]
```

416. Partition Equal Subset Sum

```python
class Solution(object):
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        total = sum(nums)
        if total%2 == 1: return False
        total /= 2
        dp = [False]*(total+1)
        dp[0] = True
        nums.sort(reverse = True)
        # top down
        for num in nums:
            for i in range(1, total+1)[::-1]:
                if i >= num:
                    # if we use num, dp[0] is True
                    dp[i] = dp[i] or dp[i-num]
                if dp[-1]: return True
        return dp[-1]
```

152. Maximum Product Subarray

```python
class Solution(object):
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # very nice code and brilliant ideas
        res = cur_min = cur_max = nums[0]
        for i in range(1, len(nums)):
            if nums[i] < 0:
                cur_min, cur_max = cur_max, cur_min
            cur_max = max(nums[i], cur_max*nums[i])
            cur_min = min(nums[i], cur_min*nums[i])
            res = max(res, cur_max)
        return res
```

392. Is Subsequence

```python
class Solution(object):
    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        i = 0
        j = 0
        cnt = 0
        while(i < len(s) and j < len(t)):
            if s[i] == t[j]: 
                cnt += 1
                i += 1
            j += 1
        if cnt == len(s): return True
        else: return False
```

63. Unique Paths II

```python
class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        n = len(obstacleGrid[0])
        dp = [0]*n
        dp[0] = 1
        for row in obstacleGrid:
            for j in range(n):
                # if 1, set it to 0 completely
                if row[j]:
                    dp[j] = 0
                # not on the sidewall
                elif j > 0:
                    dp[j] += dp[j-1]
        return dp[-1]
```

213. House Robber II

```python
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        if n == 0: return 0
        if n == 1: return nums[0]
        def helper(nums):
            pre = cur = 0
            for i in nums:
                # temp = cur
                # cur = max(cur, pre+i)
                # pre = temp
                cur, pre = max(cur, pre+i), cur
                
            return cur
        return max( helper(nums[:-1]), helper(nums[1:]) )
```

221. Maximal Square

```python
class Solution(object):
    def maximalSquare(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        # solution 3 change the dp to 1D array
        if not matrix: return 0
        m = len(matrix)
        n = len(matrix[0])
        dp = [0]*(n+1)
        res = 0 
        pre = 0
        for i in range(1,m+1):
            for j in range(1, n+1):
                temp = dp[j]
                # pre just like dp[i-1][j-1] in solution 2
                if matrix[i-1][j-1] == '1':
                    dp[j] = min(pre, dp[j-1], dp[j]) + 1
                    res = max(res, dp[j])
                else:
                    dp[j] = 0
                pre = temp
        return res*res
        # solution 2
        # if not matrix: return 0
        # m = len(matrix)
        # n = len(matrix[0])
        # dp = [[0]*(n+1) for _ in range(m+1)]
        # res = 0
        # for i in range(1, m+1):
        #     for j in range(1, n+1):
        #         if matrix[i-1][j-1] == '1':
        #             dp[i][j] = min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1])+1
        #             res = max(res, dp[i][j])
        # return res*res
#         # solution 1
#         # brute force
#         if not matrix:
#             return 0
        
#         m = len(matrix)
#         n = len(matrix[0])
#         res = 0
#         for i in range(m):
#             for j in range(n):
#                 if matrix[i][j] == '1':
#                     temp = 1
#                     flag = True
#                     while temp+i < m and temp+j < n and flag:
#                         for k in range(j, temp+j+1):
#                             if matrix[temp+i][k] == '0':
#                                 flag = False
#                                 break
#                         for k in range(i,temp+i+1):
#                             if matrix[k][temp+j] == '0':
#                                 flag = False
#                                 break
#                         if flag:
#                             temp += 1
#                     res = max(res, temp)
#         return res*res
```

62. Unique Paths

```python
class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        # grid = [[1]*n]*m
        
        # for i in xrange(1,m):
        #     for j in xrange(1,n):
        #         grid[i][j] = grid[i][j-1] + grid[i-1][j]
        
        # return grid[-1][-1]
        
        grid = [1]*m
        for j in xrange(1,n):
            for i in xrange(1,m):
                grid[i] += grid[i-1]
        return grid[-1]
```

264. Ugly Number II

```python
class Solution(object):
    def nthUglyNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        p1, p2, p3 = 0, 0, 0 #pointers in the following list
        
        q = [0] * n
        q[0] = 1
        
        for i in range(1, n):
            t1, t2, t3 = q[p1] * 2, q[p2] * 3, q[p3] * 5
            q[i] = min(t1, t2, t3)
            if q[i] == t1: p1 += 1
            if q[i] == t2: p2 += 1
            if q[i] == t3: p3 += 1
            
        return q[-1]
```

377. Combination Sum IV

```python
class Solution(object):
    def combinationSum4(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        nums.sort()
        res = [0]*(target + 1)
        res[0] = 1
        for i in range(target+1):
            for num in nums:
                if num > i: break
                res[i]+=res[i-num]
        return res[target]
```

368. Largest Divisible Subset

```python
class Solution(object):
    def largestDivisibleSubset(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # here is the compact solution
        # S = {-1: set()}
        # for x in sorted(nums):
        #     S[x] = max((S[d] for d in S if x % d == 0), key=len) | {x}
        # return list(max(S.values(), key=len))
        
        # more informative version
        # for each i, get the largest j so i%j == 0 in nums and recorded
        # for print out, just go back from end to start
        n = len(nums)
        count = [None]*n
        pre = [None]*n
        nums.sort()
        d_max = 0
        index = -1
        for i in range(n):
            count[i] = 1
            pre[i] = -1
            for j in range(i)[::-1]:
                if nums[i]%nums[j] == 0:
                    if count[i] < count[j]+1:
                        count[i] = count[j]+1
                        pre[i] = j
            if count[i] > d_max:
                d_max  = count[i]
                index = i
        res = []
        while index != -1:
            res.append(nums[index])
            index =  pre[index]
        
        return res
```

357. Count Numbers with Unique Digits

```python
class Solution(object):
    def countNumbersWithUniqueDigits(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 0: return 1
        if n >= 1 and n <= 10:
            sum = 9
            i = 1
            while(i < n):
                sum *= (10-i)
                i += 1
            return sum + self.countNumbersWithUniqueDigits(n-1)
        if n > 10:
            return self.countNumbersWithUniqueDigits(10)
```

375. Guess Number Higher or Lower II

```python
class Solution(object):
    def getMoneyAmount(self, n):
        """
        :type n: int
        :rtype: int
        """
        # state function: dp[i][j] = min(k + max(dp[i][k - 1], dp[k + 1][j]))
        if n == 1: return 0
        dp = [[0] * (n+1) for _ in range(n+1)]
        def solve(lo, hi):
            if lo >= hi: return 0
            if dp[lo][hi]: return dp[lo][hi]
            res = sys.maxint
            for x in range(lo, hi):
                temp = x + max(solve(lo,x-1), solve(x+1,hi))
                res = min(res, temp)
            dp[lo][hi] = res
            return res
        return solve(1, n)
```

343. Integer Break

```python
class Solution(object):
    def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 2: return 1
        if n == 3: return 2
        if n == 4: return 4
        if n%3 == 0: return 3**(n/3)
        if n%3 == 1: return 3**(n/3 - 1)*4
        if n%3 == 2: return 3**(n/3)*2
```

338. Counting Bits

```python
class Solution(object):
    def countBits(self, num):
        """
        :type num: int
        :rtype: List[int]
        """
        res = [0]*(num+1)
        # i = 1
        # power = 0
        # while(i<=num):
        #     if i == 2**power:
        #         res[i] = 1
        #     elif i == 2**(power+1):
        #         res[i] = 1
        #         power += 1
        #     # between two nums that are power of 2
        #     else:
        #         res[i] = 1 + res[i-2**power]
        #     i += 1
        # return res
        for i in xrange(1, num+1):
            res[i] = res[i>>1] + (i&1)
        return res
```

322. Coin Change

```python
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        # solution 2 top down
        # the maximum can not be larger than amount+1 if the minimum coin is 1
        # dp = [0] + [amount+1]*amount
        # for i in range(amount+1):
        #     for coin in coins:
        #         if i-coin >=0 and dp[i-coin] != amount+1 and dp[i-coin]+1<dp[i]:
        #             dp[i] = dp[i-coin]+1
        # return -1 if dp[amount] == amount+1 else dp[amount]
    
        # solution 1 bottom up
        # dp = [amount+1]*(amount+1)
        # dp[0] = 0
        # # coin.sort()
        # for i in range(1, amount+1):
        #     for j in range(len(coins)):
        #         if coins[j] <= i:
        #             dp[i] = min(dp[i], dp[i-coins[j]]+1)
        # return dp[amount] if dp[amount] <= amount else -1
        
                
        
        # solution 3 use bfs to calculate the tree level
        steps = collections.defaultdict(int)
        queue = collections.deque([0])
        steps[0] = 0
        coins.sort()
        while queue:
            front = queue.popleft()
            level = steps[front]
            if front == amount:
                return level
            for c in coins:
                if front + c > amount:
                    # continue
                    # if coins is sorted
                    break
                if front + c not in steps:
                    queue.append(front + c)
                    steps[front + c] = level + 1
        return -1
```

523. Continuous Subarray Sum

```python
class Solution(object):
    def checkSubarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        dmap = {0 : -1}
        total = 0
        for i, n in enumerate(nums):
            total += n
            m = total % k if k else total
            # record this index if it is not in dmap
            if m not in dmap: dmap[m] = i
            # if m in dmap and this index is larger than the previous index
            # which ensure this subsum is a multiple of k
            elif dmap[m] + 1 < i: return True
        return False
```

376. Wiggle Subsequence

```python
class Solution(object):
    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # solution 1 O(n)
        # my original thought
        # n = len(nums)
        # if n < 2: return n
        # delta = nums[1] - nums[0]
        # # as long as the first two are not the same, will attach the first two as the head
        # ans = 1 + (delta != 0)
        # for x in range(2, n):
        #     newDelta = nums[x] - nums[x-1]
        #     # not equal and the next is different from the in-place one
        #     if newDelta != 0 and newDelta * delta <= 0:
        #         ans += 1
        #         delta = newDelta
        # return ans
        
        
        # solution 2 O(n^2)
        # 利用两个辅助数组inc, dec分别保存当前状态为递增/递减的子序列的最大长度。
#         size = len(nums)
#         if size < 2: return size
        
#         inc, dec = [1] * size, [1] * size
#         for x in range(size):
#             for y in range(x):
#                 if nums[x] > nums[y]:
#                     inc[x] = max(inc[x], dec[y] + 1)
#                 elif nums[x] < nums[y]:
#                     dec[x] = max(dec[x], inc[y] + 1)
#         return max(inc[-1], dec[-1])
    
        # solution 3 change to O(n)
#         size = len(nums)
#         if size < 2: return size

#         inc, dec = [1] * size, [1] * size
#         for x in range(1, size):
#             if nums[x] > nums[x - 1]:
#                 inc[x] = dec[x - 1] + 1
#                 dec[x] = dec[x - 1]
#             elif nums[x] < nums[x - 1]:
#                 inc[x] = inc[x - 1]
#                 dec[x] = inc[x - 1] + 1
#             else:
#                 inc[x] = inc[x - 1]
#                 dec[x] = dec[x - 1]
#         return max(inc[-1], dec[-1])
    
        # solution 4 from website O(1)
        size = len(nums)
        if size < 2: return size
        
#         inc = dec = 1
#         for x in range(1, size):
#             if nums[x] > nums[x - 1]:
#                 inc = dec + 1
#             elif nums[x] < nums[x - 1]:
#                 dec = inc + 1
#         return max(inc, dec)
        # try to make short lines
        res = [1,1]
        for x in range(1, size):
            if nums[x]^nums[x-1]:
                res[nums[x]>nums[x-1]] = res[nums[x]<nums[x-1]] + 1
        return max(res)
```

309. Best Time to Buy and Sell Stock with Cooldown

```python
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        # solution 1
        # sells[i]表示在第i天卖出股票所能获得的最大累积收益
        # buys[i]表示在第i天买入股票所能获得的最大累积收益
        # 初始化令sells[0] = 0，buys[0] = -prices[0]
        # delta = price[i] - price[i - 1]
        # sells[i] = max(buys[i - 1] + prices[i], sells[i - 1] + delta) 
        # buys[i] = max(sells[i - 2] - prices[i], buys[i - 1] - delta)
        # size = len(prices)
        # if not size:
        #     return 0
        # buys = [None] * size
        # sells = [None] * size
        # sells[0], buys[0] = 0, -prices[0]
        # for x in range(1, size):
        #     delta = prices[x] - prices[x - 1]
        #     sells[x] = max(buys[x - 1] + prices[x], sells[x - 1] + delta)
        #     buys[x] = max(buys[x - 1] - delta, \
        #                   sells[x - 2] - prices[x] if x > 1 else None)
        # return max(sells)
        
#         solution 2 
#         sells[i]表示在第i天不持有股票所能获得的最大累计收益
#         buys[i]表示在第i天持有股票所能获得的最大累计收益

#         初始化数组：
#         sells[0] = 0
#         sells[1] = max(0, prices[1] - prices[0])
#         buys[0] = -prices[0]
#         buys[1] = max(-prices[0], -prices[1])
#         sells[i] = max(sells[i - 1], buys[i - 1] + prices[i])
#         buys[i] = max(buys[i - 1], sells[i - 2] - prices[i])
        
        size = len(prices)
        if size < 2:
            return 0
        buys = [None] * size
        sells = [None] * size
        sells[0], sells[1] = 0, max(0, prices[1] - prices[0])
        buys[0], buys[1] = -prices[0], max(-prices[0], -prices[1])
        for x in range(2, size):
            sells[x] = max(sells[x - 1], buys[x - 1] + prices[x])
            buys[x] = max(buys[x - 1], sells[x - 2] - prices[x])
        return sells[-1]
```

304. Range Sum Query 2D - Immutable

```python
class NumMatrix(object):

    def __init__(self, matrix):
        """
        :type matrix: List[List[int]]
        """
        if not matrix: return
        m = len(matrix)
        n = len(matrix[0])
        self.dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                self.dp[i][j] = matrix[i-1][j-1] + self.dp[i-1][j] + self.dp[i][j-1] - self.dp[i-1][j-1]
        

    def sumRegion(self, row1, col1, row2, col2):
        """
        :type row1: int
        :type col1: int
        :type row2: int
        :type col2: int
        :rtype: int
        """
        return self.dp[row2+1][col2+1] - self.dp[row1][col2+1] - self.dp[row2+1][col1] + self.dp[row1][col1]
        


# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)
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

300. Longest Increasing Subsequence

```python
class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # solution 1
        # fixed i, look back and search
        # n = len(nums)
        # if n==0:
        #     return 0
        
        # cnt = [1]*n
        # for i in xrange(n):
        #     for j in xrange(i):
        #         if nums[j] < nums[i]:
        #             # the at index j, have more increasing subarray, keep updating the counts for index i
        #             if cnt[j] + 1 > cnt[i]:
        #                 cnt[i] = cnt[j] + 1
                        
        # return max(cnt)
        
        # solution 2 using binary search
        tails = [0] * len(nums)
        size = 0
        for x in nums:
            i, j = 0, size
            while i != j:
                m = (i + j) / 2
                if tails[m] < x:
                    i = m + 1
                else:
                    j = m
            tails[i] = x
            size = max(i + 1, size)
        return size      
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