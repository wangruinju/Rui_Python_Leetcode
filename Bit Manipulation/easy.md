---
title: "Easy"
author: "Rui Wang"
date: "7/8/2017"
output: html_document
---

342. Power of Four

```python
class Solution(object):
    def isPowerOfFour(self, num):
        """
        :type num: int
        :rtype: bool
        """
        # use the fact that power of 4
        # 4^num -1 can be divided by 3
        # 2^(2n+1) -1 can not be divided by 3
        return num>0 and (num&(num-1)) == 0 and (num-1)%3 == 0
```

191. Number of 1 Bits

```python
class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        res = 0
        while n:
            # n&(n-1) remove the rightmost 1 each time
            n &= n-1
            res += 1
        return res
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

461. Hamming Distance

```python
class Solution(object):
    def hammingDistance(self, x, y):
        """
        :type x: int
        :type y: int
        :rtype: int
        """
        # solution 1
        # return bin(x^y).count('1')
        
        # solution 2
        
        res = x^y
        cnt = 0
        while res:
            cnt+=1
            # remove the rightmost 1
            res &= (res-1)
        return cnt
        
        # solution 3
        # if x^y == 0: return 0
        # return (x^y)%2 + self.hammingDistance(x>>1, y>>1)
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

405. Convert a Number to Hexadecimal

```python
class Solution(object):
    def toHex(self, num):
        """
        :type num: int
        :rtype: str
        """
        string = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
        if num == 0:
            return "0"
        if num < 0:
            num += 2**32
            
        res = ""
        while num:
            res = string[num&15] + res
            num >>= 4
        return res
```

190. Reverse Bits

```python
class Solution:
    # @param n, an integer
    # @return an integer
    def reverseBits(self, n):
        nums = [0]*32
        i = 0
        res = 0
        while n:
            nums[i] = n%2
            i += 1
            n >>= 1
        for i in xrange(32):
            res = res*2 + nums[i]

        return res
```

476. Number Complement

```python
class Solution(object):
    def findComplement(self, num):
        """
        :type num: int
        :rtype: int
        """
        # solution 1
        # flag = False
        # for i in range(32)[::-1]:
        #     # find the leftmost 1
        #     if num & (1<<i):
        #         flag =True
        #     if flag:
        #     # xor 11...1 from index i
        #         num ^= (1<<i)
        # return num
        
        # solution 2
        return (1 - num%2) + 2 * (0 if num <= 1 else self.findComplement(num>>1))
```

401. Binary Watch

```python
class Solution(object):
    def readBinaryWatch(self, num):
        """
        :type num: int
        :rtype: List[str]
        """
        res = []
        for h in xrange(12):
            for m in xrange(60):
                if (bin(h) + bin(m)).count('1') == num:
                    res.append("%d:%02d"%(h,m))
        return res
```

231. Power of Two

```python
class Solution(object):
    def isPowerOfTwo(self, n):
        """
        :type n: int
        :rtype: bool
        """
        # n&(n-1) will be 0 if n is a power of 2
        # we also need to check whether n == 0
        if n == 0: return False
        
        # return not (n&(n-1))
        while n%2==0:
            n /= 2
        return n==1
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

371. Sum of Two Integers

```python
class Solution(object):
    def getSum(self, a, b):
        """
        :type a: int
        :type b: int
        :rtype: int
        """
        # 32 bits integer max
        MAX = 0x7FFFFFFF
        # 32 bits interger min
        MIN = 0x80000000
        # mask to get last 32 bits
        mask = 0xFFFFFFFF
        while b != 0:
            # ^ get different bits and & gets double 1s, << moves carry
            a, b = (a ^ b) & mask, ((a & b) << 1) & mask
        # if a is negative, get a's 32 bits complement positive first
        # then get 32-bit positive's Python complement negative
        return a if a <= MAX else ~(a ^ mask)
```
