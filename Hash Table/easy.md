---
title: "Easy"
author: "Rui Wang"
date: "7/8/2017"
output: html_document
---

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

594. Longest Harmonious Subsequence

```python
class Solution(object):
    def findLHS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        dict = collections.Counter(nums)
        maxlength = 0
        for key in dict.keys():
            if key+1 in dict:
                maxlength = max(dict[key]+dict[key+1], maxlength)
        return maxlength
```

575. Distribute Candies

```python
class Solution(object):
    def distributeCandies(self, candies):
        """
        :type candies: List[int]
        :rtype: int
        """
        dict = {}
        cnt = 0
        for candy in candies:
            if candy not in dict:
                dict[candy] = 1
                cnt += 1
        return min(len(candies)/2, cnt)
```

500. Keyboard Row

```python
class Solution(object):
    def findWords(self, words):
        """
        :type words: List[str]
        :rtype: List[str]
        """
        list = []
        for word in words:
            temp = True
            if word.lower()[0] in 'qwertyuiop':
                for s in word.lower():
                    if temp:
                        if not s in 'qwertyuiop':
                            temp = False
            if word.lower()[0] in 'asdfghjkl':
                for s in word.lower():
                    if temp:
                        if not s in 'asdfghjkl':
                            temp = False
            if word.lower()[0] in 'zxcvbnm':
                for s in word.lower():
                    if temp:
                        if not s in 'zxcvbnm':
                            temp = False
            if temp:
                list.append(word)
        return list
```

463. Island Perimeter

```python
class Solution(object):
    def islandPerimeter(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        # solution 1
        m = len(grid)
        n = len(grid[0])
        cnt = 0
        for i in xrange(m):
            for j in xrange(n):
                if grid[i][j] == 0:
                    continue
                # if j == 0 or grid[i][j-1] == 0:
                #     cnt += 1
                # if j == n-1 or grid[i][j+1] == 0:
                #     cnt += 1
                # if i == 0 or grid[i-1][j] == 0:
                #     cnt += 1
                # if i == m-1 or grid[i+1][j] == 0:
                #     cnt += 1
                cnt += 4
                if i>0 and grid[i-1][j] == 1:
                    cnt -= 2
                if j>0 and grid[i][j-1] == 1:
                    cnt -= 2
                
        return cnt
```

447. Number of Boomerangs

```python
class Solution(object):
    def numberOfBoomerangs(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        res = 0
        distance = 0
        for i in xrange(len(points)):
            dict = {}
            for j in xrange(len(points)):
                distance = (points[i][0]-points[j][0])**2 + (points[i][1]-points[j][1])**2
                if distance in dict:
                    dict[distance] += 1
                else:
                    dict[distance] = 1
            
            for key in dict.keys():
                if dict[key] >= 2:
                    res += dict[key] * (dict[key] - 1)
        return res
```

438. Find All Anagrams in a String

```python
class Solution(object):
    def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """
        # res = []
        # if not s or not p:
        #     return res
        
        # cnt = {}
        # for _ in xrange(26):
        #     cnt[_] = 0
        
        # for str in p:
        #     cnt[ord(str) - ord('a')] += 1
    
        # for i in xrange(len(s)-len(p)+1):
        #     flag = True
        #     temp = cnt.copy()
        #     for j in xrange(i, i+len(p)):
        #         t = ord(s[j]) - ord('a')
        #         temp[t] -= 1
        #         if temp[t] < 0:
        #             flag = False
        #             break
                
        #     if flag:
        #         res.append(i)
        # return res
        
        ls, lp = len(s), len(p)
        count = lp
        cp = collections.Counter(p)
        ans = []
        for i in range(ls):
            if cp[s[i]] >=1 :
                count -= 1
            cp[s[i]] -= 1
            if i >= lp:
                if cp[s[i - lp]] >= 0:
                    count += 1
                cp[s[i - lp]] += 1
            if count == 0:
                ans.append(i - lp + 1)
        return ans
```

409. Longest Palindrome

```python
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: int
        """
        dict = {}
        for string in s:
            if string in dict:
                dict[string] += 1
            else:
                dict[string] = 1
        
        res = 0
        cnt = 0
        for key in dict.keys():
            if dict[key] %2 == 0:
                res += dict[key]
            else:
                res += dict[key] - 1
                cnt += 1
        return res + (cnt >= 1)
```

389. Find the Difference

```python
class Solution(object):
    def findTheDifference(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        # solution 1
        # translate string to value
        # then return string
        res = ord(t[len(s)])
        for i in xrange(len(s)):
            res += ord(t[i]) - ord(s[i])
        return chr(res)
        
        # solution 2 use bit manipulation
        # res = ord(t[len(s)])
        # for i in xrange(len(s)):
        #     res ^= ord(s[i])
        #     res ^= ord(t[i])
        
        # return chr(res)
        
        # solution 3 using hash
        
        # dict = {}
        # for str in s:
        #     if str in dict.keys():
        #         dict[str] += 1
        #     else:
        #         dict[str] = 1
        
        # for str in t:
        #     if str in dict.keys():
        #         if dict[str] == 0:
        #             return str
        #         else:
        #             dict[str] -= 1
        #     else:
        #         return str
```

136. Single Number

```python
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = nums[0]
        for i in xrange(1, len(nums)):
            res ^= nums[i]
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

599. Minimum Index Sum of Two Lists

```python
class Solution(object):
    def findRestaurant(self, list1, list2):
        """
        :type list1: List[str]
        :type list2: List[str]
        :rtype: List[str]
        """
        dict = {}
        res = []
        sum = len(list1) + len(list2)

        for i in xrange(len(list1)):
            dict.setdefault(list1[i], i)
        
        for j in xrange(len(list2)):
            if list2[j] in dict and j + dict[list2[j]] <= sum:
                if j + dict[list2[j]] < sum:
                    res = []
                    sum = j + dict[list2[j]]
                res.append(list2[j])
        return res
```

242. Valid Anagram

```python
class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if len(s) == len(t):
            return sorted(s) == sorted(t)
        else: return False
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

205. Isomorphic Strings

```python
class Solution(object):
    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        # solution 1
        # dict = {}
        # if len(set(s)) != len(set(t)):
        #     return False
        
        # for i in xrange(len(s)):
        #     if s[i] in dict:
        #         if dict[s[i]] != t[i]:
        #             return False
        #     else:
        #         dict[s[i]] = t[i]
        # return True
        
        d1, d2 = [0 for _ in xrange(256)], [0 for _ in xrange(256)]
        for i in xrange(len(s)):
            if d1[ord(s[i])] != d2[ord(t[i])]:
                return False
            d1[ord(s[i])] = i+1
            d2[ord(t[i])] = i+1
        return True
```

204. Count Primes

```python
class Solution(object):
    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 2:
            return 0
        primelist = [True]*n
        primelist[0:2] = [False]*2
        i = 2
        while i*i < n:
            if primelist[i]:
                primelist[i*2:n:i] = [False]*((n-1-i*2)/i+1)
            i += 1
        return sum(primelist)
```

202. Happy Number

```python
class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        dict = {}
        while True:
           dict[n] = 0
           sum = 0
           while n>0:
               sum += (n%10)*(n%10)
               n = n/10
           if sum == 1:
               return True
           elif sum in dict:
               return False
           else:
               n = sum
```

290. Word Pattern

```python
class Solution(object):
    def wordPattern(self, pattern, str):
        """
        :type pattern: str
        :type str: str
        :rtype: bool
        """
        s = str.split()
        if len(s) != len(pattern): return False
        else: return len(set(zip(s, pattern))) == len(set(s)) == len(set(pattern))
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