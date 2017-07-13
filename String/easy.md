---
title: "Easy"
author: "Rui Wang"
date: "7/8/2017"
output: html_document
---

28. Implement strStr()

```python
class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        if not needle: return 0
        m = len(haystack)
        n = len(needle)
        if m < n:
            return -1
        
        for i in xrange(m-n+1):
            j = 0
            while j < n:
                if haystack[i+j] != needle[j]:
                    break
                j += 1
            if j == n:
                return i
        return -1
```

557. Reverse Words in a String III

```python
class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        strs = s.split(' ')
        for i in range(len(strs)):
            strs[i] = strs[i][::-1]
        return ' '.join(strs)
```

551. Student Attendance Record I

```python
class Solution(object):
    def checkRecord(self, s):
        """
        :type s: str
        :rtype: bool
        """
        # cheat
        # return not (s.count('A') >=2 or 'LLL' in s)
        cnt = 0
        repeat = 0
        for string in s:
            if string  == 'A':
                cnt += 1
                repeat = 0
            elif string == 'L':
                repeat += 1
            else:
                repeat = 0
            if cnt > 1 or repeat > 2:
                return False
                
        return True
```

541. Reverse String II

```python
class Solution(object):
    def reverseStr(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: str
        """
        new = ''
        t = len(s)
        res = t%(2*k)
        n = t/(2*k)
        if n > 0:
            for i in range(n):
                new += self.reverse(s[k*2*i:k*(2*i+1)])
                new += s[k*(2*i+1):k*2*(i+1)]
        if res <= k:
            new += self.reverse(s[k*2*n:])
        else:
            new += self.reverse(s[k*2*n:k*(2*n+1)])
            new += s[k*(2*n+1):]
        return new
    def reverse(self, s):
        return s[::-1]
        
        ## short answer
        # s = list(s)
        # for i in range(0,len(s),2*k):
        #     s[i:i+k] = reversed(s[i,i+k])
        # return ''.join(s)
```

521. Longest Uncommon Subsequence I

```python
class Solution(object):
    def findLUSlength(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: int
        """
        if a == b: return -1
        return max(len(a), len(b))
```

13. Roman to Integer

```python
class Solution(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        roman = {'M': 1000,'D': 500 ,'C': 100,'L': 50,'X': 10,'V': 5,'I': 1}
        z = 0
        for i in range(0, len(s) - 1):
            if roman[s[i]] < roman[s[i+1]]:
                z -= roman[s[i]]
            else:
                z += roman[s[i]]
        return z + roman[s[-1]]
```

14. Longest Common Prefix

```python
class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if len(strs) == 0: return ''
        pre = strs[0]
        for i in xrange(1, len(strs)):
            while strs[i].find(pre) != 0:
                pre = pre[:-1]
        return pre
```

520. Detect Capital

```python
class Solution(object):
    def detectCapitalUse(self, word):
        """
        :type word: str
        :rtype: bool
        """
        return word.capitalize() == word or word.upper() == word or word.lower() == word
```

20. Valid Parentheses

```python
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        for string in s:
            if string in "([{":
                stack.append(string)
            if string == ")":
                if not stack or stack.pop() != "(":
                    return False
            if string == "]":
                if not stack or stack.pop() != "[":
                    return False
            if string == "}":
                if not stack or stack.pop() != "{":
                    return False
        
        return False if stack else True
```

459. Repeated Substring Pattern

```python
class Solution(object):
    def repeatedSubstringPattern(self, s):
        """
        :type s: str
        :rtype: bool
        """
        m = len(s)
        n = m/2
        while(n >= 1):
            if m%n == 0:
                if s[:n]*(m/n) == s: return True
            n -= 1
        return False
```

606. Construct String from Binary Tree

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def tree2str(self, t):
        """
        :type t: TreeNode
        :rtype: str
        """
        
        if not t:
            return ''
        elif not t.left and not t.right:
            return str(t.val)
        elif not t.left and t.right:
            return str(t.val) + "(" + ")" + "(" + self.tree2str(t.right) + ")"
        elif t.left and not t.right:
            return str(t.val) + "(" + self.tree2str(t.left) + ")"
        else:
            return str(t.val) + "(" + self.tree2str(t.left) + ")" + "(" + self.tree2str(t.right) + ")"
```

434. Number of Segments in a String

```python
class Solution(object):
    def countSegments(self, s):
        """
        :type s: str
        :rtype: int
        """
        cnt = 0
        flag = True
        for i in range(len(s)):
            if s[i] != ' ' and flag:
                cnt += 1
                flag = False
            if s[i] == ' ':
                flag = True
        return cnt   
```

38. Count and Say

```python
class Solution(object):
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        res = '1'
        for i in range(1,n):
            list = ''
            temp = res[0]
            cnt = 1
            for j in range(1,len(res)):
                if res[j] == temp:
                    cnt += 1
                else:
                    list += str(cnt) + temp
                    temp = res[j]
                    cnt = 1
            list += str(cnt) + temp
            res = list 
        return res
```

383. Ransom Note

```python
class Solution(object):
    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        # use collections.Counter
        # if works, it should be empty dict like Counter()
        # in opposite, it should be ture
        return not collections.Counter(ransomNote) - collections.Counter(magazine) 
```

345. Reverse Vowels of a String

```python
class Solution(object):
    def reverseVowels(self, s):
        """
        :type s: str
        :rtype: str
        """
        l, r = 0, len(s)-1
        ls = list(s)
        while l < r:
            while l < r and not ls[l].lower() in "aeiou":
                l += 1
            while l < r and not ls[r].lower() in "aeiou":
                r -= 1
            temp = ls[l]
            ls[l] = ls[r]
            ls[r] = temp
            l += 1
            r -= 1
        return ''.join(ls)
```

344. Reverse String

```python
class Solution(object):
    def reverseString(self, s):
        """
        :type s: str
        :rtype: str
        """
        l = list(s)
        # l.reverse()
        n = len(l)
        for i in range(n/2):
            l[i], l[n-1-i] = l[n-1-i], l[i]
        return ''.join(l)        
```

58. Length of Last Word

```python
class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        
        n2 = len(s) - 1
        while(n2 >= 0 and s[n2] == ' '):
            n2 -= 1
        n1 = n2
        while(n1 >= 0 and s[n1] != ' '):
            n1 -= 1
        return n2-n1
```

67. Add Binary

```python
class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        # return bin(int(a,2) + int(b,2))[2:]
        if len(a)==0: return b
        if len(b)==0: return a
        if a[-1] == '1' and b[-1] == '1':
            return self.addBinary(self.addBinary(a[:-1],b[:-1]),'1')+'0'
        if a[-1] == '0' and b[-1] == '0':
            return self.addBinary(a[:-1],b[:-1])+'0'
        else:
            return self.addBinary(a[:-1],b[:-1])+'1'
```

125. Valid Palindrome

```python
class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        
        l, r = 0, len(s)-1
        while l < r:
            while l < r and not s[l].isalnum():
                l += 1
            while l < r and not s[r].isalnum():
                r -= 1
            if s[l].lower() != s[r].lower():
                return False
            l += 1
            r -= 1
        return True
```
