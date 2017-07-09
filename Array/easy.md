---
title: "Easy"
author: "Rui Wang"
date: "7/8/2017"
output: html_document
---

268. Missing Number

```python
class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # solution 1
        # use math 
        # return len(nums)*(len(nums)+1)/2 - sum(nums)
        
        # solution 2
        # use bit manipulation
        res = 0
        for i in xrange(len(nums)+1):
            res ^= i
        
        for n in nums:
            res ^= n
        return res
```

1. Two Sum

```python
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # for i in range(len(nums)-1):
        #      if target - nums[i] in nums[i+1:]:
        #          return [i, i+1+(nums[i+1:]).index(target - nums[i])]
        # return []
        
        idxDict = dict()
        for idx, num in enumerate(nums):
            if target - num in idxDict:
                return [idxDict[target - num], idx]
            idxDict[num] = idx
```

283. Move Zeroes

```python
class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        j = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[j] = nums[i]
                j += 1
        for i in range(j,len(nums)):
            nums[i] = 0
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

169. Majority Element

```python
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = nums[0]
        cnt = 1
        for i in xrange(1, len(nums)):
            if cnt == 0:
                cnt += 1
                res = nums[i]
            elif res == nums[i]:
                cnt += 1
            else:
                cnt -= 1
        return res
```

189. Rotate Array

```python
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        # solution 1
        # for i in range(k):
        #     nums.insert(0, nums.pop())
        
        # solution 2
        # n = len(nums)
        # if k > 0 and n > 1:
        #     nums[:] = nums[n - k:] + nums[:n - k]
        
        # solution 3
#         n = len(nums)
#         k %= n
#         self.reverse(nums, 0, n - k)
#         self.reverse(nums, n - k, n)
#         self.reverse(nums, 0, n)

#     def reverse(self, nums, start, end):
#         for x in range(start, (start + end) / 2):
#             nums[x] ^= nums[start + end - x - 1]
#             nums[start + end - x - 1] ^= nums[x]
#             nums[x] ^= nums[start + end - x - 1]

        # solution 4
        # numscopy = nums[:]
        # n = len(nums)
        # for i in range(n):
        #     nums[(i+k)%n] = numscopy[i]
        
        # solution 5
        n = len(nums)
        idx = 0
        distance = 0
        cur = nums[0]
        for x in range(n):
            idx = (idx + k) % n
            nums[idx], cur = cur, nums[idx]
            
            distance = (distance + k) % n
            if distance == 0:
                idx = (idx + 1) % n
                cur = nums[idx]
```

26. Remove Duplicates from Sorted Array

```python
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        i = 0
        for num in nums:
            if i < 1 or num > nums[i-1]:
                nums[i] = num
                i += 1
        return i
```

27. Remove Element

```python
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        cnt = 0
        for num in nums:
            if num != val:
                nums[cnt] = num
                cnt += 1
        return cnt
```

561. Array Partition I

```python
class Solution(object):
    def arrayPairSum(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort()
        sum = 0
        for i in range(len(nums)/2):
            sum += nums[i*2]
        return sum
```

605. Can Place Flowers

```python
class Solution(object):
    def canPlaceFlowers(self, flowerbed, n):
        """
        :type flowerbed: List[int]
        :type n: int
        :rtype: bool
        """
        # solution 1 get the maximum number and compare it with n
#         num = len(flowerbed)
#         index_one = []
#         for i, flower in enumerate(flowerbed):
#             if flower:
#                 index_one.append(i)

#         if not index_one:
#             return (num+1)/2 >= n
#         res = index_one[0]/2 + (num-1-index_one[-1])/2
#         for i in range(len(index_one)-1):
#             res += (index_one[i+1] - index_one[i] - 2)/2
#         return res >= n


        # solution 2 greedy 
        # count = 0
        # i = 0
        # num = len(flowerbed)
        # while i < num and count < n:
        #     if flowerbed[i] == 0:
        #         next = 0 if i == num-1 else flowerbed[i+1]
        #         pre = 0 if i == 0 else flowerbed[i-1]
        #         if next == 0 and pre == 0:
        #             flowerbed[i] = 1
        #             count += 1
        #     i += 1
        # return count == n
        
        # same idea but concise codes
        num = len(flowerbed)
        for i in xrange(num):
            if (flowerbed[i] == 0 and (i == 0 or flowerbed[i-1] == 0) 
                    and (i == len(flowerbed)-1 or flowerbed[i+1] == 0)):
                n -= 1
                flowerbed[i] = 1
        return n <= 0
```

217. Contains Duplicate

```python
class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if nums == []: return False
        dict = {}
        for num in nums:
            if not num in dict:
                dict[num] = 1
            else:
                return True
        return False
        # or len(set(nums)) != len(nums) kind of cheat
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

448. Find All Numbers Disappeared in an Array

```python
class Solution(object):
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        for i in range(len(nums)):
            index = abs(nums[i]) - 1
            # not change the value, will use the index again so use absolute
            nums[index] = - abs(nums[index])
        return [i+1 for i in range(len(nums)) if nums[i] > 0]
```

66. Plus One

```python
class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        carry = 1
        res = []
        for i in range(len(digits))[::-1]:
            res.append((digits[i]+carry)%10)
            carry = (digits[i] + carry)/10
        if carry: res.append(carry)
        return res[::-1]
```

485. Max Consecutive Ones

```python
class Solution(object):
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        count = 0
        temp = 0
        for num in nums:
            if num == 1:
                count += 1
                temp = max(temp, count)
            else:
                count = 0
        return temp
```

581. Shortest Unsorted Continuous Subarray

```python
class Solution(object):
    def findUnsortedSubarray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        # -1 and -2 just initialization
        start = -1
        end = -2
        # set min and max flip at two ends
        list_min = nums[n-1]
        list_max = nums[0]
        
        for i in range(1, n):
            list_max = max(list_max, nums[i])
            list_min = min(list_min, nums[n-1-i])
            
            if nums[i] < list_max: end = i
            if nums[n-1-i] > list_min: start = n-1-i
        return end - start + 1
```

414. Third Maximum Number

```python
class Solution(object):
    def thirdMax(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # if len(set(nums)) < 3: return max(nums)
        # else:
        #     return sorted(set(nums), reverse = True)[2]
        
        a = b = c = None
        for n in nums:
            if n > a:
                a, b, c = n, a, b
            elif a > n > b:
                b, c = n, b
            elif b > n > c:
                c = n
        return c if c is not None else a
```

122. Best Time to Buy and Sell Stock II

```python
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        res = 0
        for i in range(1, len(prices)):
            if prices[i] - prices[i-1] > 0:
                res += prices[i] - prices[i-1]
        return res
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

566. Reshape the Matrix

```python
class Solution(object):
    def matrixReshape(self, nums, r, c):
        """
        :type nums: List[List[int]]
        :type r: int
        :type c: int
        :rtype: List[List[int]]
        """
        m = len(nums[0])
        n = len(nums)
        if m*n != r*c:
            return nums
            
        res = itertools.chain(*nums)
        return [list(itertools.islice(res, c)) for i in xrange(r)]
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

119. Pascal's Triangle II

```python
class Solution(object):
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        p = [1]
        if rowIndex == 0: return p
        for i in range(rowIndex):
            p = map(lambda x, y: x+y, [0]+p, p+[0])
        return p
```

118. Pascal's Triangle

```python
class Solution(object):
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        p = [[1]]
        for i in range(1,numRows):
            p += [map(lambda x, y: x+y, p[-1] + [0], [0] + p[-1])]
        return p[:numRows]
```

628. Maximum Product of Three Numbers

```python
class Solution(object):
    def maximumProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort()
        return max(nums[-1]*nums[-2]*nums[-3], nums[0]*nums[1]*nums[-1])
```

532. K-diff Pairs in an Array

```python
class Solution(object):
    def findPairs(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        if k > 0:
            return len(set(nums) & set(num + k for num in nums))
        elif k == 0:
            return sum(v > 1 for v in collections.Counter(nums).values())
        else:
            return 0
```

219. Contains Duplicate II

```python
class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        dict = {}
        for i,v in enumerate(nums):
            if v in dict and i - dict[v] <= k:
                return True
            dict[v] = i
        return False
```

88. Merge Sorted Array

```python
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        while m > 0 and n > 0:
            if nums1[m-1] >= nums2[n-1]:
                nums1[m+n-1] = nums1[m-1]
                m -= 1
            else:
                nums1[m+n-1] = nums2[n-1]
                n -= 1
        if n > 0:
            nums1[:n] = nums2[:n]
```



