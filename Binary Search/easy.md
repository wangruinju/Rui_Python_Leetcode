---
title: "Easy"
author: "Rui Wang"
date: "7/8/2017"
output: html_document
---

367. Valid Perfect Square

```python
class Solution(object):
    def isPerfectSquare(self, num):
        """
        :type num: int
        :rtype: bool
        """
        # same problem to get int(sqrt(num))
        # use Newton's method
        res = num
        while(res * res > num):
            res = (res + num/res)/2
        return res*res == num
```

167. Two Sum II - Input array is sorted

```python
class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        ind1 = 0
        ind2 = len(numbers) - 1
        while(numbers[ind1] + numbers[ind2] != target):
            if numbers[ind1] + numbers[ind2] > target:
                ind2 -= 1
            else:
                ind1 += 1
        return [ind1+1, ind2+1]
```

441. Arranging Coins

```python
class Solution(object):
    def arrangeCoins(self, n):
        """
        :type n: int
        :rtype: int
        """
        # solution 1
        # i = 1
        # while (i+1)*i <= 2*n:
        #     i += 1
        # return i-1
    
        # solution 2
        # return int( ( (1.0+8.0*n)**0.5 - 1)/2.0)
        
        # solution 3
        start = 0
        end = n
        while start <= end:
            mid = start + (end-start)/2
            if mid*(mid+1)/2 <= n:
                start = mid+1
            else:
                end = mid-1
        return start-1
```

35. Search Insert Position

```python
class Solution(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        for i in range(len(nums)):
            if nums[i] >= target:
                return i
        return len(nums)
```

374. Guess Number Higher or Lower

```python
# The guess API is already defined for you.
# @param num, your guess
# @return -1 if my number is lower, 1 if my number is higher, otherwise return 0
# def guess(num):

class Solution(object):
    def guessNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        low = 1
        high = n
        while(low < high):
            mid = (low + high) / 2
            if guess(mid) == 1:
                low = mid + 1
            else:
                high = mid
        return low
```

69. Sqrt(x)

```python
class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        r = x
        while r*r > x:
            r = (r + x/r) / 2
        return r
```

278. First Bad Version

```python
# The isBadVersion API is already defined for you.
# @param version, an integer
# @return a bool
# def isBadVersion(version):

class Solution(object):
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        left = 1
        right = n
        while left < right:
            mid = (left+right)/2
            if not isBadVersion(mid):
                left = mid + 1
            else:
                right = mid
        return left
```

475. Heaters

```python
class Solution(object):
    def findRadius(self, houses, heaters):
        """
        :type houses: List[int]
        :type heaters: List[int]
        :rtype: int
        """
        # Based on 2 pointers, the idea is to find the nearest heater for each house, 
        # by comparing the next heater with the current heater.


        heaters.sort()
        houses.sort()
        i = res = 0
        for house in houses:
            while i < len(heaters)-1 and heaters[i]+heaters[i+1] <= house*2:
                i += 1
            res = max(res, abs(heaters[i] - house))
        return res
```

350. Intersection of Two Arrays II

```python
class Solution(object):
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        # solution 1 use collectins.Couter
        # a = collections.Counter(nums1) & collections.Counter(nums2)
        # return list(a.elements())
    
        # solution 2 use two pointer
        nums1.sort()
        nums2.sort()
        i = 0
        j = 0
        res = []
        while i < len(nums1) and j < len(nums2):
            if nums1[i] == nums2[j]:
                res.append(nums1[i])
                i += 1
                j += 1
            elif nums1[i] > nums2[j]:
                j += 1
            else:
                i += 1
        return res
```

349. Intersection of Two Arrays

```python
class Solution(object):
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        # solution 1
        # return list(set(nums1) & set(nums2))
        
        # solution 2
        
        dict1 = {}
        for num in nums1:
            dict1[num] = dict1.get(num, 1)
            
        dict2 = {}
        for num in nums2:
            if num in dict1:
                dict2[num] = dict2.get(num, 1)
        
        res = []
        for key in dict2.keys():
            res.append(key)
        return res
    
        # use binary search 
        # search (for num in nums1) in nums2
```