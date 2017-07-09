---
title: "Medium"
author: "Rui Wang"
date: "7/8/2017"
output: html_document
---

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

495. Teemo Attacking

```python
class Solution(object):
    def findPoisonedDuration(self, timeSeries, duration):
        """
        :type timeSeries: List[int]
        :type duration: int
        :rtype: int
        """
        if not timeSeries or not duration: return 0
        res = 0
        n = len(timeSeries)
        for i in range(1, n):
            diff = timeSeries[i] - timeSeries[i-1]
            res += diff if diff < duration else duration
        return res + duration
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

621. Task Scheduler

```python
class Solution(object):
    def leastInterval(self, tasks, n):
        """
        :type tasks: List[str]
        :type n: int
        :rtype: int
        """
        
        dict = collections.Counter(tasks)
        count = 0
        maxcnt = 0
        for value in dict.values():
            if value > maxcnt:
                count = 0
                maxcnt = value
            if value == maxcnt:
                count += 1
        # if this arrangement does not work, use the total length and it will fit
        # for example: AABBC -> ABABC for n = 1 case
        return max(len(tasks), (maxcnt-1)*(n+1) + count)
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

90. Subsets II

```python
class Solution(object):
    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        self.res = []
        def dfs(nums, temp, index):
            self.res.append(temp[:])
            for i in range(index, len(nums)):
                # add conditions to remove duplicates
                if i>index and nums[i-1]==nums[i]:
                    continue
                temp.append(nums[i])
                dfs(nums, temp, i+1)
                temp.pop()
        dfs(nums, [], 0)
        return self.res
```

81. Search in Rotated Sorted Array II

```python
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: bool
        """
        # edge case
        n = len(nums)
        if n == 0:
            return False
        
        # binary search
        left = 0
        right = n-1
        while left <= right:
            mid = (left+right)/2
            if nums[mid] == target:
                return True
            # right part is ascending
            elif nums[mid] < nums[right]:
                if nums[mid] < target and target <= nums[right]:
                    left = mid+1
                else:
                    right = mid-1
            # left part is ascending
            elif nums[mid] > nums[right]:
                if nums[left] <= target and target < nums[mid]:
                    right = mid-1
                else:
                    left = mid+1
            else:
                right -= 1
        return False
```

79. Word Search

```python
class Solution(object):
    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        # 使用dfs来搜索，为了避免已经用到的字母被重复搜索，将已经用到的字母临时替换为'*'就可以了。
        def dfs(i, j, word):
            if len(word) == 0: return True
            
            if i > 0 and board[i-1][j] == word[0]:
                # use '*' to mark i,j that we have temporarily applied
                temp = board[i][j]
                board[i][j] = '*'
                if dfs(i-1, j, word[1:]):
                    return True
                board[i][j] = temp
            
            if i < len(board)-1 and board[i+1][j] == word[0]:
                temp = board[i][j]
                board[i][j] = '*'
                if dfs(i+1, j, word[1:]):
                    return True
                board[i][j] = temp
                
            if j > 0 and board[i][j-1] == word[0]:
                temp = board[i][j]
                board[i][j] = '*'
                if dfs(i, j-1, word[1:]):
                    return True
                board[i][j] = temp
                
            if j < len(board[0])-1 and board[i][j+1] == word[0]:
                temp = board[i][j]
                board[i][j] = '*'
                if dfs(i, j+1, word[1:]):
                    return True
                board[i][j] = temp
            
            
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == word[0]:
                    if dfs(i, j, word[1:]):
                        return True
        return False
```

78. Subsets

```python
class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        self.res = []
        def dfs(nums, temp, index):
            self.res.append(temp[:])
            for i in range(index, len(nums)):
                temp.append(nums[i])
                dfs(nums, temp, i+1)
                temp.pop()
            
        dfs(nums, [], 0)
        return self.res
```

75. Sort Colors

```python
class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        left = 0
        right = len(nums) - 1
        for i in range(len(nums)):
            while(nums[i] == 2 and i < right):
                temp = nums[i]
                nums[i] = nums[right]
                nums[right] = temp
                right -= 1
            while(nums[i] == 0 and i > left):
                temp = nums[i]
                nums[i] = nums[left]
                nums[left] = temp
                left += 1
```

74. Search a 2D Matrix

```python
class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if not matrix:
            return False
        
        n = len(matrix)
        m = len(matrix[0])
        l = 0
        r = m*n-1
        while l <= r:
          mid = (l+r)/2
          temp = matrix[mid/m][mid%m]
          if temp > target:
            r = mid - 1
          elif temp < target:
            l = mid + 1
          else:
            return True
        return False
```

73. Set Matrix Zeroes

```python
class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        # solution 2
        # point to a object
#         obj = []
         
#         for i in range(len(matrix)):
#             for j in range(len(matrix[0])):
#                 if matrix[i][j] == 0:
#                     for k in range(len(matrix)):
#                         if matrix[k][j] != 0:
#                             matrix[k][j] = obj
#                     for k in range(len(matrix[0])):
#                         if matrix[i][k] != 0:
#                             matrix[i][k] = obj
         
#         for i in range(len(matrix)):
#             for j in range(len(matrix[0])):
#                 if matrix[i][j] is obj:
#                     matrix[i][j] = 0
        # solution 1
        flag = True
        m = len(matrix)
        n = len(matrix[0])
        
        for i in range(m):
            if matrix[i][0] == 0:
                flag = False
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = matrix[0][j] = 0
        
        for i in range(m)[::-1]:
            for j in range(1, n)[::-1]:
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
            if not flag:
                matrix[i][0] = 0
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

59. Spiral Matrix II

```python
class Solution(object):
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        # solution 2
        # smart solution
        # rotate the -1
        A = [[0] * n for _ in range(n)]
        i, j, di, dj = 0, 0, 0, 1
        for k in xrange(n*n):
            A[i][j] = k + 1
            if A[(i+di)%n][(j+dj)%n]:
                di, dj = dj, -di
            i += di
            j += dj
        return A
    
#         solution 1 based on previous spiral matrix I
#         if n == 0:
#             return []
        
#         matrix = [[0]*n for _ in range(n)]
        
#         r1 = 0
#         r2 = n-1
#         c1 = 0
#         c2 = n-1
#         k = 1
        
#         while r1 <= r2 and c1 <= c2:
#             # traverse right
#             for j in range(c1, c2+1):
#                 matrix[r1][j] = k
#                 k += 1
#             r1 += 1
            
#             # traverse down
#             for j in range(r1, r2+1):
#                 matrix[j][c2] = k
#                 k += 1
#             c2 -= 1
            
#             if r1 <= r2:
#                 # traverse left
#                 for j in range(c1, c2+1)[::-1]:
#                     matrix[r2][j] = k
#                     k += 1
#             r2 -= 1
            
#             if c1 <= c2:
#                 # traverse up
#                 for j in range(r1, r2+1)[::-1]:
#                     matrix[j][c1] = k
#                     k += 1
#             c1 += 1
#         return matrix
```

560. Subarray Sum Equals K

```python
class Solution(object):
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        # count = collections.Counter()
        # count[0] = 1
        # cnt = 0
        # sum = 0
        # for n in nums:
        #     sum += n
        #     cnt += count[sum-k]
        #     count[sum] += 1
        # return cnt
        count = {0:1}
        cnt = 0
        sum = 0
        for n in nums:
            sum += n
            cnt += count.get(sum-k, 0)
            count[sum] = count.get(sum, 0) + 1
        return cnt
```

56. Merge Intervals

```python
# Definition for an interval.
# class Interval(object):
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        res = []
        for i in sorted(intervals, key = lambda i : i.start):
        # if overlap happens: i.start <= previous.end
            if res and i.start <= res[-1].end:
                res[-1].end = max(res[-1].end, i.end)
            else:
                res.append(i)
        return res
```

55. Jump Game

```python
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        # top down method
        # jump back from the last index to the start
        n = len(nums)
        last = n-1
        for i in xrange(2, n+1):
            # if n-i position can jump to the last
            # set last = n-i
            
            if nums[n-i] + n-i >= last:
                last = n-i
                
        return last <= 0
```

54. Spiral Matrix

```python
class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
    
        # solution 1
        # cut the top, right, bottom, left in order
        # such a smart solution
        # ret = []
        # while matrix:
        #     ret += matrix.pop(0)
        #     if matrix and matrix[0]:
        #         for row in matrix:
        #             ret.append(row.pop())
        #     if matrix:
        #         ret += matrix.pop()[::-1]
        #     if matrix and matrix[0]:
        #         for row in matrix[::-1]:
        #             ret.append(row.pop(0))
        # return ret
        
        # solution 2
        res = []
        if not matrix:
            return res
        
        r1 = 0
        r2 = len(matrix)-1
        c1 = 0
        c2 = len(matrix[0])-1
        
        while r1 <= r2 and c1 <= c2:
            # traverse right
            for j in range(c1, c2+1):
                res.append(matrix[r1][j])
            r1 += 1
            
            # traverse down
            for j in range(r1, r2+1):
                res.append(matrix[j][c2])
            c2 -= 1
            
            if r1 <= r2:
                # traverse left
                for j in range(c1, c2+1)[::-1]:
                    res.append(matrix[r2][j])
            r2 -= 1
            
            if c1 <= c2:
                # traverse up
                for j in range(r1, r2+1)[::-1]:
                    res.append(matrix[j][c1])
            c1 += 1
    
        return res
```

48. Rotate Image

```python
class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        if n == 0 or n == 1:
            return 
        
        # flip from digonal
        for i in range(n-1):
            for j in range(n-1-i):
                matrix[i][j], matrix[n-1-j][n-1-i] = matrix[n-1-j][n-1-i], matrix[i][j]
        
        # flip from up down
        for i in range(n/2):
            for j in range(n):
                matrix[i][j], matrix[n-1-i][j] = matrix[n-1-i][j], matrix[i][j]
```

565. Array Nesting

```python
class Solution(object):
    def arrayNesting(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # follow the rules in the questions and write the code
        # use -1 to mark that it has been visited
        maxsize = 0
        for i in range(len(nums)):
            size = 0
            k = i
            while nums[k] >= 0:
                temp = nums[k]
                nums[k] = -1
                k = temp
                size += 1
            
            maxsize = max(maxsize, size)
            
        return maxsize
```

611. Valid Triangle Number

```python
class Solution(object):
    def triangleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort()
        count = 0
        n = len(nums)
        for i in range(2,n)[::-1]:
            left = 0
            right = i-1
            while left < right:
                if nums[left] + nums[right] > nums[i]:
                    count += right - left
                    right -= 1
                else:
                    left += 1
        return count
```

40. Combination Sum II

```python
class Solution(object):
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        self.res = []
        candidates.sort()
        
        def dfs(nums, temp, remainder, start):
            if remainder == 0:
                self.res.append(temp[:])
                return
            if remainder < 0:
                return
            
            for i in xrange(start, len(nums)):
                # condition for duplicate
                # after start we will skip the case
                if i>start and nums[i-1] == nums[i]:
                    continue
                
                temp.append(nums[i])
                dfs(nums, temp, remainder - nums[i], i+1)
                temp.pop()

        dfs(candidates, [], target, 0)
        return self.res
```

39. Combination Sum

```python
class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        self.res = []

        def dfs(nums, temp, remainder, index):
            if remainder == 0:
                self.res.append(temp[:])
                return
            
            if remainder < 0:
                return
            
            for i in xrange(index, len(nums)):
                temp.append(nums[i])
                dfs(nums, temp, remainder - nums[i], i)
                temp.pop()
        
        dfs(candidates, [], target, 0)
        
        return self.res
```

34. Search for a Range

```python
class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        n = len(nums)
        res = [-1, -1]
        if not n:
            return res
        
        l = 0
        r = n-1
        while l < r:
            m = (l+r)/2
            if nums[m] < target:
                l = m+1
            else:
                r = m
        if nums[l] != target:
            return res
        res[0] = l
        
        r = n-1
        while l < r:
            m = (l+r)/2+1
            if nums[m] > target:
                r = m-1
            else:
                l = m
        res[1] = r
        return res
```

33. Search in Rotated Sorted Array

```python
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        # edge case
        n = len(nums)
        if n == 0:
            return -1
        
        # binary search
        left = 0
        right = n-1
        while left <= right:
            mid = (left+right)/2
            if nums[mid] == target:
                return mid
            # right part is ascending
            elif nums[mid] < nums[right]:
                if nums[mid] < target and target <= nums[right]:
                    left = mid+1
                else:
                    right = mid-1
            # left part is ascending
            else:
                if nums[left] <= target and target < nums[mid]:
                    right = mid-1
                else:
                    left = mid+1
        return -1
```

31. Next Permutation

```python
class Solution(object):
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        size = len(nums)
        # backwards to find the first index
        for first in range(size - 1, -1, -1):
            if nums[first - 1] < nums[first]:
                break
        if first > 0:
            # find the second index and swap with the first-1 index
            for second in range(size - 1, -1, -1):
                if nums[second] > nums[first - 1]:
                    nums[first - 1], nums[second] = nums[second], nums[first - 1]
                    break
        # reverse between first and end
        for i in range((size - first) / 2):
            nums[first + i], nums[size - i - 1] = nums[size - i - 1], nums[first + i]
```

18. 4Sum

```python
class Solution(object):
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        n = len(nums)
        res = []
        if n < 4: 
            return res
        nums.sort()
        
        for i in range(n-3):
            if i > 0 and nums[i]==nums[i-1]:
                continue
            if nums[i]+nums[i+1]+nums[i+2]+nums[i+3]>target: 
                break
            if nums[i]+nums[n-3]+nums[n-2]+nums[n-1]<target:
                continue
        
            for j in range(i+1, n-2):
                if j > i+1 and nums[j]==nums[j-1]:
                    continue
                if nums[i]+nums[j]+nums[j+1]+nums[j+2]>target:
                    break
                if nums[i]+nums[j]+nums[n-2]+nums[n-1]<target:
                    continue
                
                left = j+1; right = n-1
                while(left < right):
                    temp = nums[i]+nums[j]+nums[left]+nums[right]
                    if temp < target:
                        left += 1
                    elif temp > target:
                        right -= 1
                    else:
                        res.append([nums[i], nums[j], nums[left], nums[right]])
                        while(left < right and nums[left]==nums[left+1]):
                            left += 1
                        while(left < right and nums[right]==nums[right-1]):
                            right -= 1
                        left += 1; right -= 1
                    
        return res
```

16. 3Sum Closest

```python
class Solution(object):
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        n = len(nums)
        nums.sort()
        # initialize res with the first three numbers
        res = nums[0] + nums[1] + nums[2]
        for i in range(n-2):
            left = i+1; right = n-1
            while left < right:
                sum = nums[i] + nums[left] + nums[right]
                if sum == target:
                    return sum
                if abs(sum - target) < abs(res - target):
                    res = sum
                if sum < target:
                    left += 1
                elif sum > target:
                    right -= 1
        return res
```

15. 3Sum

```python
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        n = len(nums)
        res = []
        for i in range(n-2):
            # get a unique start
            if i > 0 and nums[i] == nums[i-1]:
                continue
            left = i+1; right = n-1
            while left < right:
                target = nums[i] + nums[left] + nums[right]
                if target < 0:
                    left += 1
                elif target > 0:
                    right -= 1
                else:
                    # if matched, attach the result
                    res.append([nums[i], nums[left], nums[right]])
                    # remove the duplicate
                    while left < right and nums[left] == nums[left+1]:
                        left += 1
                    while left < right and nums[right] == nums[right-1]:
                        right -= 1
                    # next search
                    left += 1; right -= 1
        return res
```

11. Container With Most Water

```python
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        left = 0
        right = len(height) - 1
        # intial the water as 0
        res = 0
        while (left < right):
            h = min(height[left], height[right])
            res = max(res, h*(right-left))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return res
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

80. Remove Duplicates from Sorted Array II

```python
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        i = 0
        for num in nums:
            # skip the first k = 2 elements, move if num > nums[i-k]
            if i < 2 or num > nums[i-2]:
                nums[i] = num
                i += 1
        return i
```

380. Insert Delete GetRandom O(1)

```python
class RandomizedSet(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.datamap = {}
        self.datalist = []

    def insert(self, val):
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        if val in self.datamap:
            return False
        else:
            self.datamap[val] = len(self.datalist)
            self.datalist.append(val)
            return True
        

    def remove(self, val):
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        :type val: int
        :rtype: bool
        """
        if val not in self.datamap:
            return False
        else:
            index = self.datamap[val]
            tail = self.datalist.pop()
            # if the val is not the tail of datalist
            # if the tail corresponding to the index, we have already removed it
            if index < len(self.datalist):
                self.datalist[index] = tail
                self.datamap[tail] = index
            del self.datamap[val]
            
            return True
        


    def getRandom(self):
        """
        Get a random element from the set.
        :rtype: int
        """
        return random.choice(self.datalist)
        


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()
```

289. Game of Life

```python
class Solution(object):
    def gameOfLife(self, board):
        """
        :type board: List[List[int]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        
        dx = [-1, 0,  1, -1, 1, -1, 0, 1]
        dy = [-1, -1, -1, 0, 0,  1, 1, 1]
        m = len(board)
        n = len(board[0])
        for i in range(m):
            for j in range(n):
                cnt = 0
                for k in xrange(8):
                    x = i + dx[k]
                    y = j + dy[k]
                    
                    if x < 0 or x >= len(board) or y < 0 or y >= len(board[0]):
                        continue
                    
                    if board[x][y] == 1 or board[x][y] == 2:
                            cnt += 1
                
                if board[i][j] == 1 and (cnt < 2 or cnt > 3):
                    board[i][j] = 2
                elif board[i][j] == 0 and cnt == 3:
                    board[i][j] = 3
        
        for i in range(m):
            for j in range(n):
                board[i][j] %= 2
        
    
        # smart ideas
        # use %2 to get the updated status
        # 由于细胞只有两种状态0和1，因此可以使用二进制来表示细胞的生存状态
        # 更新细胞状态时，将细胞的下一个状态用高位进行存储
        # 全部更新完毕后，将细胞的状态右移一位


#         dx = (1, 1, 1, 0, 0, -1, -1, -1)
#         dy = (1, 0, -1, 1, -1, 1, 0, -1)
#         for x in range(len(board)):
#             for y in range(len(board[0])):
#                 lives = 0
#                 for z in range(8):
#                     nx, ny = x + dx[z], y + dy[z]
#                     lives += self.getCellStatus(board, nx, ny)
#                     # very smart codes
#                     # lives = 3 board = 0 -> board = 2
#                     # lives = 3 board = 1 -> board = 3
#                     # lives = 2 board = 1 -> board = 3
#                 if lives + board[x][y] == 3 or lives == 3:
#                     board[x][y] |= 2
#         for x in range(len(board)):
#             for y in range(len(board[0])):
#                 board[x][y] >>= 1

#     def getCellStatus(self, board, x, y):
#         if x < 0 or y < 0 or x >= len(board) or y >= len(board[0]):
#             return 0
#         return board[x][y] & 1
```

287. Find the Duplicate Number

```python
class Solution(object):
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        slow = 0
        fast = 0
        while(True):
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break
        fast = 0
        while(True):
            slow = nums[slow]
            fast = nums[fast]
            if slow == fast:
                return slow
```

238. Product of Array Except Self

```python
class Solution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        n = len(nums)
        res = [0]*n
        # solution 1
#         res[0] = 1
#         for i in range(1,n):
#             res[i] = res[i-1]*nums[i-1]
        
#         right = 1
#         for i in range(n)[::-1]:
#             res[i] *= right
#             right *= nums[i]
        
#         return res
        # solution 2 like this code symmetry
        temp = 1
        for i in range(n):
            res[i] = temp
            temp *= nums[i]
        temp = 1
        for i in range(n)[::-1]:
            res[i] *= temp
            temp *= nums[i]
        return res
```

229. Majority Element II

```python
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # solution 1
        # res = []
        # dict = collections.Counter(nums)
        # for key in dict.keys():
        #     if dict[key] > len(nums)/3:
        #         res.append(key)
        # return res
        
        if not nums:
            return []
        
        count1, count2, candidate1, candidate2 = 0, 0, 0, 1
        for n in nums:
            if n == candidate1:
                count1 += 1
            elif n == candidate2:
                count2 += 1
            elif count1 == 0:
                candidate1, count1 = n, 1
            elif count2 == 0:
                candidate2, count2 = n, 1
            else:
                count1, count2 = count1 - 1, count2 - 1
        return [n for n in (candidate1, candidate2) if nums.count(n) > len(nums)/3]
```

228. Summary Ranges

```python
class Solution(object):
    def summaryRanges(self, nums):
        """
        :type nums: List[int]
        :rtype: List[str]
        """
        res = []
        i = 0
        while i < len(nums):
            temp = nums[i]
            while i < len(nums)-1 and nums[i+1] - nums[i] == 1:
                i += 1
            if temp != nums[i]:
                res.append(str(temp) + "->" + str(nums[i]))
            else:
                res.append(str(temp))
            i += 1
                
        return res
```

216. Combination Sum III

```python
class Solution(object):
    def combinationSum3(self, k, n):
        """
        :type k: int
        :type n: int
        :rtype: List[List[int]]
        """
        # same like combination I and II
        # no duplicate and from 1 to 9
        # be careful with the variables when you define some functions

        self.res = []
        
        def dfs(temp, remainder, k, start):
            if remainder == 0 and len(temp) == k:
                self.res.append(temp[:])
                return 
            if remainder < 0:
                return
            
            for i in xrange(start, 10):
                temp.append(i)
                dfs(temp, remainder-i, k, i+1)
                temp.pop()
                
        dfs([], n, k, 1)
        return self.res
```

209. Minimum Size Subarray Sum

```python
class Solution(object):
    def minSubArrayLen(self, s, nums):
        """
        :type s: int
        :type nums: List[int]
        :rtype: int
        """
        # solution 1 using O(n)
        # if not nums:
        #     return 0
        # left = 0
        # right = 0
        # sum = 0
        # res = len(nums)+1
        # while right < len(nums):
        #     while sum < s and right < len(nums):
        #         sum += nums[right]
        #         right += 1
        #     while sum >= s:
        #         res = min(res, right-left)
        #         sum -= nums[left]
        #         left += 1
        
        # return res if res != len(nums)+1 else 0
        
        res = len(nums)+1
        left = 0
        sum = 0
        for i in xrange(len(nums)):
            sum += nums[i]
            while left <= i and sum >= s:
                res = min(res, i-left+1)
                sum -= nums[left]
                left += 1
        return res if res != len(nums)+1 else 0
        
        # solution 2 using binary search
    #     n = len(nums)
    #     sums = [0]*(n+1)
    #     for i in xrange(1, n+1):
    #         sums[i] = sums[i-1] + nums[i-1]
        
    #     minLen = n+1
    #     for i in xrange(n+1):
    #         # sums[j] satisfy sums[j] >= sums[i] + s
    #         # the length is j - i
    #         end = self.helper(i+1, n, sums[i]+s, sums)
    #         # end is out of the arrays, no subarray qualify
    #         if end == n+1: break
    #         minLen = min(minLen, end - i)
    #     return minLen if minLen != n+1 else 0
    
    # def helper(self, lo, hi, key, sums):
    #     while lo <= hi:
    #         mid = (hi + lo)/2
    #         if sums[mid] >= key:
    #             hi = mid - 1
    #         else:
    #             lo = mid + 1
    #     return lo
```

162. Find Peak Element

```python
class Solution(object):
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # solution 1 sequential
        # for i in range(1,len(nums)):
        #     if nums[i] < nums[i-1]:
        #         return i-1
        # return len(nums)-1
    
        # solution 2 binary search
        l = 0
        h = len(nums)-1
        while l < h:
            m1 = (l+h)/2
            if nums[m1] < nums[m1+1]:
                l = m1+1
            else:
                h = m1
        return l
```

442. Find All Duplicates in an Array

```python
class Solution(object):
    def findDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        res = []
        for num in nums:
            if nums[abs(num) - 1] < 0:
                res.append(abs(num))
            else:
                nums[abs(num) - 1] *= -1
        return res
```

153. Find Minimum in Rotated Sorted Array

```python
class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        left = 0
        right = n-1
        while left < right:
            if nums[left] < nums[right]:
                return nums[left]
            
            mid = (left+right)/2
            if nums[mid] >= nums[left]:
                left = mid+1
            else:
                right = mid
        
        return nums[left]
```