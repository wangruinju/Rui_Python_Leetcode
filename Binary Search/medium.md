---
title: "Medium"
author: "Rui Wang"
date: "7/8/2017"
output: html_document
---

436. Find Right Interval

```python
# Definition for an interval.
# class Interval(object):
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution(object):
    def findRightInterval(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[int]
        """
        # solution 1
        # use bisect
        # invs = sorted((x.start, i) for i, x in enumerate(intervals))
        # ans = []
        # for x in intervals:
        #     # Find rightmost index whose value is greater than or equal to x.end
        #     idx = bisect.bisect_right( invs, (x.end,) )
        #     # if the index is out set it to -1
        #     ans.append(invs[idx][1] if idx < len(intervals) else -1)
        # return ans
        
        # solution 2
        # write the binary code into the for loop
    
        sorted_start = [(interval.start, index) for (index, interval) in enumerate(intervals)]
        sorted_start.sort()
        result = []

        for interval in intervals:
            end = interval.end
            lo = 0
            hi = len(intervals)
            while lo < hi:
                mid = (lo + hi) // 2
                if sorted_start[mid][0] < end:
                    lo = mid+1
                else:
                    hi = mid
            if lo == len(intervals):
                result.append(-1)
            else:
                result.append(sorted_start[lo][1])

        return result
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

275. H-Index II

```python
class Solution(object):
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """
        # recall the definition of h-index
        # A scientist has index h if h of his/her N papers have at least h citations each, 
        # and the other N − h papers have no more than h citations each.
        if not citations:
            return 0
        n = len(citations)
        l = 0
        r = n-1
        
        while l <= r:
            m = (l+r)/2
            if n-m == citations[m]:
                return citations[m]
            elif n-m < citations[m]:
                r = m-1   
            else:
                l = m+1
        return n-r-1
```

240. Search a 2D Matrix II

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        # solution 1 
        # use the property of bst
        # take all elements in a array orderly
        # return the kth element
        # it take a lot of spaces if the tree has too many elements
    #     res = []
    #     self.helper(root, res)
    #     return res[k-1]
    
    # def helper(self, node, res):
    #     if not node:
    #         return 
    #     self.helper(node.left, res)
    #     res.append(node.val)
    #     self.helper(node.right, res)
    
        # solution 2
        # binary search 
        # repeatedly count the tree on the left
    #     count = self.count(root.left)
    #     # if left count is larger or equal than k, pass to the left node
    #     if count >= k:
    #         return self.kthSmallest(root.left, k)
    #     # count+1 left+current node
    #     elif count+1 < k:
    #         return self.kthSmallest(root.right, k-count-1)
    #     # if count+1 == k
    #     return root.val
        
    # def count(self, root):
    #     if not root:
    #         return 0
    #     return 1+self.count(root.left)+self.count(root.right)
        
        # solution 3
        # dfs in order: left root right
        # self.res = 0
        # self.count = k
        # def dfs(root):
        #     if not root:
        #         return 0
        #     dfs(root.left)
        #     self.count -=1
        #     if self.count == 0:
        #         self.res = root.val
        #         return
        #     dfs(root.right)
    
        # dfs(root)
        # return self.res
        
        # solution 4
        # use stack
        stack = []
        
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            k -= 1
            if k == 0: 
                break
            root = root.right
        return root.val
```

378. Kth Smallest Element in a Sorted Matrix

```python
class Solution(object):
    def kthSmallest(self, matrix, k):
        """
        :type matrix: List[List[int]]
        :type k: int
        :rtype: int
        """
        lo = matrix[0][0]
        hi = matrix[-1][-1]
        
        while lo < hi:
            mid = lo + (hi-lo)/2
            cnt = 0
            j = len(matrix[0])-1
            for i in range(len(matrix)):
                while j >= 0 and matrix[i][j] > mid:
                    j -= 1
                # for i th row: index 0 to j are <= mid
                cnt += (j+1)
            if cnt < k: 
                lo = mid+1
            else:
                hi = mid
        return hi
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

454. 4Sum II

```python
class Solution(object):
    def fourSumCount(self, A, B, C, D):
        """
        :type A: List[int]
        :type B: List[int]
        :type C: List[int]
        :type D: List[int]
        :rtype: int
        """
        ans = 0
        cnt = {}
        for a in A:
            for b in B:
                cnt[a + b] = cnt.get(a + b, 0) + 1
        for c in C:
            for d in D:
                ans += cnt.get(-(c + d), 0)
        return ans
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

50. Pow(x, n)

```python
class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n==0:
            return 1
        elif n<0:
            return 1/self.myPow(x, -n)
        else:
            half = self.myPow(x,n/2)
            if (n%2) == 0:
                return half*half
            else:
                return half*half*x
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

29. Divide Two Integers

```python
class Solution(object):
    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """
        sign = (dividend < 0) ^ (divisor < 0)
        dividend, divisor = abs(dividend), abs(divisor)
        res = 0
        while dividend >= divisor:
            temp, i = divisor, 1
            while dividend >= temp:
                dividend -= temp
                res += i
                i <<= 1
                temp <<= 1
        if sign:
            res = -res
        return min(max(-2147483648, res), 2147483647)
```

222. Count Complete Tree Nodes

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def countNodes(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # O(N) is out of time
        # if not root: return 0
        # return 1 + self.countNodes(root.left) + self.countNodes(root.right)
        
        if not root: return 0
        left = right = root
        height = 0
        while right:
            left = left.left
            right = right.right
            height += 1
        if not left:
            # 1 2 4 ... 2^(height-1) the sum is 2^height-1
            return (1 << height) - 1
        return 1 + self.countNodes(root.left) + self.countNodes(root.right)
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
