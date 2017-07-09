---
title: "Medium"
author: "Rui Wang"
date: "7/8/2017"
output: html_document
---

397. Integer Replacement

```python
class Solution(object):
    def integerReplacement(self, n):
        """
        :type n: int
        :rtype: int
        """
        # solution 1
        # if n == 1: return 0
        # if n%2 == 0: return 1 + self.integerReplacement(n>>1)
        # else:
        #     return 2 + min(self.integerReplacement((n-1)>>1), self.integerReplacement((n+1)>>1))
        
        # solution 2
        cnt = 0
        while n > 1:
            cnt += 1
            # if n is odd
            if n%2 == 0:
                n >>= 1
            else:
                # if the two rightmost bits is 11
                # exclude the edge case n == 3
                if n&2 and n!=3:
                    n += 1
                else: 
                    n -= 1
        return cnt
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

318. Maximum Product of Word Lengths

```python
class Solution(object):
    def maxProduct(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        # solution 1
        # 用了mask，因为题目中说都是小写字母，那么只有26位，一个整型数int有32位，
        # 我们可以用后26位来对应26个字母，若为1，说明该对应位置的字母出现过,
        # 那么每个单词的都可由一个int数字表示，两个单词没有共同字母的条件是这两个int数想与为0
        # res = 0
        # n = len(words)
        # mask = [0]*n
        
        # for i in xrange(n):
        #     for str in words[i]:
        #         mask[i] |= 1 << (ord(str) -ord('a'))
                
        #     for j in xrange(i):
        #         if not mask[i]&mask[j]:
        #             res = max(res, len(words[i])*len(words[j]))
            
        # return res
        
        # solution 2
        # 借助哈希表，映射每个mask的值和其单词的长度，每算出一个单词的mask，
        # 遍历哈希表里的值，如果和其中的mask值相与为0
        # 则将当前单词的长度和哈希表中存的单词长度相乘并更新结果
        
        res = 0
        dict = {}
        
        for word in words:
            mask = 0
            for string in word:
                mask |= 1 << (ord(string) - ord('a'))
                
            if mask in dict:
                dict[mask] = max(dict[mask], len(word))
            else:
                dict[mask] = len(word)
                
            for key in dict.keys():
                if not key&mask:
                    res = max(res, len(word)*dict[key])
        return res
```

260. Single Number III

```python
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # solution 1
        
        diff = 0
        for n in nums:
            diff ^= n
        # get the last bit
        diff &= -diff
        
        
        res = [0, 0]
        for n in nums:
            # if the last bit is not set
            if n&diff == 0:
                res[0] ^= n
            # if the last bit is set
            else:
                res[1] ^= n
        return res
        
        # solution 2
        # using hash table
        # dict = {}
        # for n in nums:
        #     if n in dict:
        #         dict[n] += 1
        #     else:
        #         dict[n] = 1
        # res = []
        # for key in dict.keys():
        #     if dict[key] == 1:
        #         res.append(key)
        # return res
```

393. UTF-8 Validation

```python
class Solution(object):
    def validUtf8(self, data):
        """
        :type data: List[int]
        :rtype: bool
        """
        # for i in range(len(data)):
        #     if data[i] < 0b10000000:
        #         continue
        #     else:
        #         cnt = 0
        #         val = data[i]
        #         for j in range(1,8)[::-1]:
        #             if val >= 2**j:
        #                 cnt += 1
        #             else: 
        #                 break
        #             val -= 2**j
        #         if cnt == 1: return False
        #         for j in range(i+1, i+cnt):
        #             if data[j] > 0b10111111 or data[j] < 0b10000000:
        #                 return False
        #         i += cnt-1
        # return True
        
        cnt = 0
        for d in data:
            if cnt == 0:
                if d >> 5 == 0b110:
                    cnt = 1
                elif d >> 4 == 0b1110:
                    cnt = 2
                elif d >> 3 == 0b11110:
                    cnt = 3
                elif d >> 7:
                    return False
            else:
                if d >> 6 != 0b10:
                    return False
                cnt -= 1
        return cnt == 0
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

201. Bitwise AND of Numbers Range

```python
class Solution(object):
    def rangeBitwiseAnd(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        # count the bits we move to the right
        # cnt = 0
        # while m != n:
        #     m >>= 1
        #     n >>= 1
        #     cnt += 1
        # return m << cnt
        
        # same idea
        if m == 0:
            return 0
        move = 1
        while m != n:
            m >>= 1
            n >>= 1
            move <<= 1
        return m*move
```

187. Repeated DNA Sequences

```python
class Solution(object):
    def findRepeatedDnaSequences(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        
        seen = {}
        res = []
        for i in xrange(len(s)-9):
            string  = s[i:i+10]
            if string in seen:
                seen[string] += 1
            else:
                seen[string] = 1
        
        for string in seen.keys():
            if seen[string] > 1:
                res.append(string)
        return res
```

477. Total Hamming Distance

```python
class Solution(object):
    def totalHammingDistance(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        res = 0
        for i in range(32)[::-1]:
            cnt = 0
            for num in nums:
                if num & (1<<i):
                    cnt += 1
            res += (n-cnt)*cnt
        return res
```

421. Maximum XOR of Two Numbers in an Array

```python
class Solution(object):
    def findMaximumXOR(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # since our interest is on the maximum of XOR
        # use a mask from left to right
        
        res = 0
        mask = 0
        for i in range(32)[::-1]:
            mask |= (1 << i)
            s = {}
            # reserve Left bits and ignore Right bits
            for num in nums:
                s.setdefault(num&mask)
            # Use 0 to keep the bit, 1 to find XOR
            # in each iteration, there are pair(s) whoes Left bits can XOR to max
            
            temp = res | (1 << i)
            for prefix in s.keys():
                if (temp^prefix) in s:
                    # only care one pair works, that's good enough
                    # keep that bit
                    # for example
                    # s = ---1--- and ---0--- such pair exist
                    # use ---1--- xor to check whether we have such pair in the hash
                    res = temp
                    break
            
        return res
```

137. Single Number II

```python
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # solution 1
        # ones = twos = 0
        # for n in nums:
        #     ones = (ones^n) & ~twos
        #     twos = (twos^n) & ~ones
        # return ones
        
        # solution 2
        dict = {}
        for n in nums:
            if n in dict:
                dict[n] += 1
            else:
                dict[n] = 1
        for n in nums:
            if dict[n] == 1:
                return n
```