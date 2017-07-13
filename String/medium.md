---
title: "Medium"
author: "Rui Wang"
date: "7/8/2017"
output: html_document
---

22. Generate Parentheses

```python
class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        res = []
        if n == 0:
            return res
            
        self.helpler(n, n, '', res)
        return res
        
    def helpler(self, l, r, item, res):
        if r < l:
            return
        if l == 0 and r == 0:
            res.append(item)
        if l > 0:
            self.helpler(l - 1, r, item + '(', res)
        if r > 0:
            self.helpler(l, r - 1, item + ')', res)
```

635. Design Log Storage System

```python
class LogSystem(object):

    def __init__(self):
        self.logs = []
        

    def put(self, id, timestamp):
        """
        :type id: int
        :type timestamp: str
        :rtype: void
        """
        self.logs.append((id, timestamp))
        

    def retrieve(self, s, e, gra):
        """
        :type s: str
        :type e: str
        :type gra: str
        :rtype: List[int]
        """
        index = {'Year': 5, 'Month': 8, 'Day': 11, 
                 'Hour': 14, 'Minute': 17, 'Second': 20}[gra]
        start = s[:index]
        end = e[:index]
        
        return [id for id, timestamp in self.logs if start <= timestamp[:index] <= end]


# Your LogSystem object will be instantiated and called as such:
# obj = LogSystem()
# obj.put(id,timestamp)
# param_2 = obj.retrieve(s,e,gra)
```

583. Delete Operation for Two Strings

```python
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        n1 = len(word1)
        n2 = len(word2)
        dp = [[0]*(n2+1) for _ in xrange(n1+1)]
        
        for i in xrange(n1+1):
            for j in xrange(n2+1):
                if i == 0 or j == 0:
                    dp[i][j] = 0
                elif word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return n1 + n2 - 2*dp[n1][n2]
```

385. Mini Parser

```python
# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """
#class NestedInteger(object):
#    def __init__(self, value=None):
#        """
#        If value is not specified, initializes an empty list.
#        Otherwise initializes a single integer equal to value.
#        """
#
#    def isInteger(self):
#        """
#        @return True if this NestedInteger holds a single integer, rather than a nested list.
#        :rtype bool
#        """
#
#    def add(self, elem):
#        """
#        Set this NestedInteger to hold a nested list and adds a nested integer elem to it.
#        :rtype void
#        """
#
#    def setInteger(self, value):
#        """
#        Set this NestedInteger to hold a single integer equal to value.
#        :rtype void
#        """
#
#    def getInteger(self):
#        """
#        @return the single integer that this NestedInteger holds, if it holds a single integer
#        Return None if this NestedInteger holds a nested list
#        :rtype int
#        """
#
#    def getList(self):
#        """
#        @return the nested list that this NestedInteger holds, if it holds a nested list
#        Return None if this NestedInteger holds a single integer
#        :rtype List[NestedInteger]
#        """

class Solution(object):
    def deserialize(self, s):
        """
        :type s: str
        :rtype: NestedInteger
        """
        def nestedInteger(x):
            if isinstance(x, int):
                return NestedInteger(x)
            lst = NestedInteger()
            for y in x:
                lst.add(nestedInteger(y))
            return lst
        return nestedInteger(eval(s))
```

43. Multiply Strings

```python
class Solution(object):
    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        # return str(int(num1)*int(num2))
        
        
        # edge case
        if num1 == '0' or num2 == '0':
            return '0'
            
        m = len(num1)
        n = len(num2)
        
        pos = [0]*(m+n)
        
        for i in range(m)[::-1]:
            for j in range(n)[::-1]:
                temp = (ord(num1[i]) - ord('0'))*(ord(num2[j]) - ord('0'))
                # consider the carry
                sum = temp + pos[i+j+1]
                pos[i+j] += sum/10
                pos[i+j+1] = sum%10
        res = ''
        for p in pos:
            if p != 0 or len(res) != 0:
                res += str(p)
        return res
```

49. Group Anagrams

```python
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        dict = {}
        for str in strs:
            temp = ''.join(sorted(str))
            if temp not in dict:
                dict[temp] = [str]
            else:
                dict[temp].append(str)

        res = []
        for key in dict.keys():
            res += [dict[key]]
            
        return res
```

6. ZigZag Conversion

```python
class Solution(object):
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        
        res = ['']*numRows
        i = 0
        while i < len(s):
            for j in range(numRows):
                if i < len(s):
                    res[j] += s[i]
                    i += 1
            for j in range(1, numRows-1)[::-1]:
                if i < len(s):
                    res[j] += s[i]
                    i += 1
        ans = ''
        for i in range(numRows):
            ans += res[i]
        return ans
```

556. Next Greater Element III

```python
class Solution(object):
    def nextGreaterElement(self, n):
        """
        :type n: int
        :rtype: int
        """
        # same as next permutation
        nums = list(str(n))
        size = len(nums)
        for first in range(size-1, -1, -1):
            if nums[first] > nums[first-1]:
                break
        if first > 0:
            for second in range(size-1, -1, -1):
                if nums[second] > nums[first-1]:
                    nums[second], nums[first-1] = nums[first-1], nums[second]
                    break
        
        for i in range((size-first)/2):
            nums[first+i], nums[size-1-i] = nums[size-1-i], nums[first+i]
        
        res = int(''.join(nums))
        if res > n and res < (1<<31):
            return res
        else:
            return -1
```

553. Optimal Division

```python
class Solution(object):
    def optimalDivision(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        if len(nums) == 1:
            return str(nums[0])
        if len(nums) == 2:
            return "%s/%s"%(nums[0], nums[1])
        res = ''
        for i in xrange(1, len(nums)):
            res += str(nums[i]) + "/"
        return str(nums[0]) + "/" + "(" + res[:-1] + ")"
```

227. Basic Calculator II

```python
class Solution(object):
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s: return 0
        
        num = 0
        digits = '0123456789'
        operator = '+-*/'
        sign = '+'
        stack = []
        for i in range(len(s)):
            if s[i] in digits:
                num = num*10 + int(s[i])
            if s[i] in operator or i == len(s)-1:
                if sign == '+':
                    stack.append(num)
                if sign == '-':
                    stack.append(-num)
                if sign == '/':
                    stack.append(int(stack.pop()/float(num)))
                if sign == '*':
                    stack.append(stack.pop()*num)
                # reset num
                num = 0
                sign = s[i]
        
        res = 0
        for i in stack:
            res += i
        return res
```

8. String to Integer (atoi)

```python
class Solution(object):
    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        # set up flag allow the calculation of results to run only one time for ' ' '+' '-'
        # if happen another time, stop and show the current results
        flag = 0
        sign = 1
        results = 0
        for s in str:
            if '0' <= s <= '9':
                results = results * 10 + int(s)
                flag = 1
            elif s == ' ' and flag == 0:
                sign = 1

            elif s == '+' and flag == 0:
                sign = 1
                flag = 1
            elif s == '-' and flag == 0:
                sign = -1
                flag = 1
            else:
                break
        results = results * sign
        if results > 2 ** 31 - 1:
            return 2 ** 31 - 1
        elif results < -2 ** 31:
            return -2 ** 31
        else:
            return results
```

165. Compare Version Numbers

```python
class Solution(object):
    def compareVersion(self, version1, version2):
        """
        :type version1: str
        :type version2: str
        :rtype: int
        """
        list1 = version1.split('.')
        list2 = version2.split('.')
        n1 = len(list1)
        n2 = len(list2)
        max_length = max(n1, n2)
        for i in xrange(max_length):
            v1 = int(list1[i]) if i < len(list1) else 0
            v2 = int(list2[i]) if i < len(list2) else 0
            if v1 > v2:
                return 1
            elif v1 < v2:
                return -1
        return 0
```

609. Find Duplicate File in System

```python
class Solution(object):
    def findDuplicate(self, paths):
        """
        :type paths: List[str]
        :rtype: List[List[str]]
        """
        M = collections.defaultdict(list)
        for line in paths:
            data = line.split()
            root = data[0]
            for file in data[1:]:
                name, _, content = file.partition('(')
                M[content[:-1]].append(root + '/' + name)
                
        return [x for x in M.values() if len(x) > 1]
```

539. Minimum Time Difference

```python
class Solution(object):
    def findMinDifference(self, timePoints):
        """
        :type timePoints: List[str]
        :rtype: int
        """
        res = [int(time.split(':')[0])*60 + int(time.split(':')[1]) for time in timePoints]
        
        res.sort()
        time_min = 1440+res[0]-res[-1]
        for i in range(len(res)-1):
            time_min = min(time_min, res[i+1]-res[i])
        return time_min
```

537. Complex Number Multiplication

```python
class Solution(object):
    def complexNumberMultiply(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        # ind1 = a.find('+')
        # ind2 = a.find('i')
        # ind3 = b.find('+')
        # ind4 = b.find('i')
        # a1 = int(a[:ind1])
        # a2 = int(a[ind1 + 1:ind2])
        # b1 = int(b[:ind3])
        # b2 = int(b[ind3 + 1:ind4])
        # return str(a1*b1-a2*b2) + '+' + str(a1*b2+a2*b1) + 'i'
        a1, a2 = map(int, a[:-1].split('+'))
        b1, b2 = map(int, b[:-1].split('+'))
        return '%d+%di' % (a1 * b1 - a2 * b2, a1 * b2 + a2 * b1)
```

151. Reverse Words in a String

```python
class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        list = s.split()
        list = list[::-1]
        return ' '.join(list)
```

71. Simplify Path

```python
class Solution(object):
    def simplifyPath(self, path):
        """
        :type path: str
        :rtype: str
        """
        stack = []
        for p in path.split("/"):
          if p == "..":
            if stack: stack.pop()
          elif p and p != '.': stack.append(p)
        return "/" + "/".join(stack)
```

5. Longest Palindromic Substring

```python
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        if len(s) == 0:
            return ''
        maxlength = 1
        start = 0
        for i in xrange(len(s)):
            # add every character and check whether the maxlength increase by two or one
            # it can't increase by three otherwise it will violate the maxlength
            if i-maxlength >= 1 and s[i-maxlength-1:i+1] == s[i-maxlength-1:i+1][::-1]:
                start = i-maxlength-1
                maxlength += 2
                continue
            if i-maxlength >= 0 and s[i-maxlength:i+1] == s[i-maxlength:i+1][::-1]:
                start = i-maxlength
                maxlength += 1
        return s[start:start+maxlength]
```

522. Longest Uncommon Subsequence II

```python
class Solution(object):
    def findLUSlength(self, strs):
        """
        :type strs: List[str]
        :rtype: int
        """
        strs.sort(key = len, reverse = True)
        for i, word1 in enumerate(strs):
            if all(not self.subseq(word1, word2) for j, word2 in enumerate(strs) if i != j):
                return len(word1)
        return -1
    
    def subseq(self, w1, w2):
        # check if w1 is a subseq of w2
        i = 0
        for c in w2:
            if i < len(w1) and w1[i] == c:
                i += 1
        return i == len(w1)
```

93. Restore IP Addresses

```python
class Solution(object):
    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        res = []
        n = len(s)
        i = 1
        while i < 4 and i < n-2:
            j = i+1
            while j < i+4 and j < n-1:
                k = j+1
                while k < j+4 and k < n:
                    if self.valid(s[0:i]) and self.valid(s[i:j]) and self.valid(s[j:k]) and self.valid(s[k:]):
                        res.append(s[0:i] + '.' + s[i:j] + '.' + s[j:k] + '.' + s[k:])
                    k += 1
                j += 1    
            i += 1
        return res
    def valid(self, string):
        if len(string) > 3 or len(string) == 0 or (string[0] == '0' and len(string) > 1) or int(string) > 255:
            return False
        return True
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

12. Integer to Roman

```python
class Solution(object):
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        if num >= 1000:
            return 'M'*(num/1000) + self.intToRoman(num%1000)
        elif num >= 900:
            return 'CM' + self.intToRoman(num-900)
        elif num >= 500:
            return 'D' + 'C'*((num-500)/100) + self.intToRoman(num%100)
        elif num >= 400:
            return 'CD' + self.intToRoman(num-400)
        elif num >= 100:
            return 'C'*(num/100) + self.intToRoman(num%100)
        elif num >= 90:
            return 'XC' + self.intToRoman(num-90)
        elif num >= 50:
            return 'L' + 'X'*((num-50)/10) + self.intToRoman(num%10)
        elif num >= 40:
            return 'XL' + self.intToRoman(num-40)
        elif num >= 10:
            return 'X'*(num/10) + self.intToRoman(num%10)
        elif num == 9:
            return 'IX'
        elif num >= 5:
            return 'V' + (num-5)*'I'
        elif num == 4:
            return 'IV'
        else:
            return 'I'*num
```

17. Letter Combinations of a Phone Number

```python
class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if not digits:
            return []
        list = ["0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"]
        res = [""]
        for i in range(len(digits)):
            x = int(digits[i])
            while len(res[-1]) == i:
                temp = res.pop()
                for string in list[x]:
                    res.insert(0, temp+string)
        return res
```

468. Validate IP Address

```python
class Solution(object):
    def validIPAddress(self, IP):
        """
        :type IP: str
        :rtype: str
        """
        a = self.IPv4(IP)
        b = self.IPv6(IP)
        if a:
            return "IPv4"
        if b:
            return "IPv6"
        return "Neither"
    
    def IPv4(self, IP):
        if IP.count('.') == 3:
            res = IP.split('.')
            for string in res:
                if len(string) > 0 and len(string) <= 4:
                    for s in string:
                        if s not in "0123456789":
                            return False
                    if len(string) > 1 and string[0] == '0' or int(string) >= 256:
                        return False
                else:
                    return False
                
            return True
        else:
            return False
    
    def IPv6(self, IP):
        if IP.count(':') == 7:
            res = IP.split(':')
            for string in res:
                if len(string) > 0 and len(string) <= 4:
                    for s in string:
                        if s not in "0123456789abcdefABCDEF":
                            return False
                else:
                    return False
            return True
        else:
            return False
```

3. Longest Substring Without Repeating Characters

```python
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        start = 0
        end = 0
        dict = {}
        for i in range(len(s)):
            if s[i] in dict and start <= dict[s[i]]:
                start = dict[s[i]] + 1
            else:
                end = max(end, i-start+1)
            dict[s[i]] = i
        return end
```