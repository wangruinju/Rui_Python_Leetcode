---
title: "Easy"
author: "Rui Wang"
date: "7/8/2017"
output: html_document
---

303. Range Sum Query - Immutable

```python
class NumArray(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.sum = [0]
        temp = 0
        for n in nums:
            temp += n
            self.sum.append(temp)


    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        return self.sum[j+1]-self.sum[i]
        
# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(i,j)
```

53. Maximum Subarray

```python
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        maxcur = maxsofar = nums[0]

        for i in xrange(1, len(nums)):
            maxcur = max(maxcur+nums[i], nums[i])
            maxsofar = max(maxsofar, maxcur)
        return maxsofar
```

198. House Robber

```python
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # a = 0
        # b = 0
        # for i in xrange(len(nums)):
        #     if i%2 == 0:
        #         a = max(a+nums[i], b)
        #     else:
        #         b = max(a, b+nums[i])
                
        # return max(a, b)
        last, now = 0, 0
        
        for i in nums: last, now = now, max(last + i, now)
                
        return now
```

121. Best Time to Buy and Sell Stock

```python
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        maxend = 0
        maxsofar = 0
        for i in range(1, len(prices)):
            maxend = max(maxend + prices[i] - prices[i-1], 0)
            maxsofar = max(maxend, maxsofar)
        return maxsofar
```

70. Climbing Stairs

```python
class Solution(object):
    def __init__(self):
        self.hash = {1:1, 2:2}

    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        # solution 1 bottom up
        # a, b = 1, 1
        # for i in xrange(n):
        #     a, b = b, a+b
        # return a
        
        # solution 2
        
        if n not in self.hash:
            self.hash[n] = self.climbStairs(n-1) + self.climbStairs(n-2)
        return self.hash[n]
```

