---
title: "Medium"
author: "Rui Wang"
date: "7/8/2017"
output: html_document
---

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

46. Permutations

```python
class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        # solution 1
        
        # self.res = []
        # self.used = [False]*len(nums)
        # def dfs(nums, temp):
        #     if len(temp) ==  len(nums):
        #         self.res.append(temp[:])
            
        #     for i in xrange(len(nums)):
    
        #         if self.used[i]:
        #             continue
        #         self.used[i] = True
        #         temp.append(nums[i])
        #         dfs(nums, temp)
        #         temp.pop()
        #         self.used[i] = False
        # dfs(nums, [])
        # return self.res
        
        # solution 2
        # start from the first number
        # add one to every existing list at different positions
        ans = [[]]   
        for n in nums:
            new_ans = []
            for list in ans:
                for i in xrange(len(list)+1):   
                    new_ans.append(list[:i] + [n] + list[i:])   ###insert n
            ans = new_ans
        return ans
```

47. Permutations II

```python
class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        #
        # nums.sort()
        # self.res = []
        # self.used = [False]*len(nums)
        # def dfs(nums, temp):
        #     if len(nums) == len(temp):
        #         self.res.append(temp[:])
                
        #     for i in xrange(len(nums)):
        #         if self.used[i]: continue
            
        #         if i>0 and nums[i] == nums[i-1] and self.used[i-1]:
        #             continue
        #         self.used[i] = True
        #         temp.append(nums[i])
        #         dfs(nums, temp)
        #         temp.pop()
        #         self.used[i] = False
        # dfs(nums, [])
        # return self.res
        
        
        # solution 2
        ans = [[]]
        for n in nums:
            new_ans = []
            for list in ans:
                for i in xrange(len(list)+1):
                    new_ans.append(list[:i] + [n] + list[i:])
                    # we don't attach the duplicate after the same value
                    if i<len(list) and list[i] == n: break
            ans = new_ans
        return ans
```

526. Beautiful Arrangement

```python
class Solution(object):
    def countArrangement(self, N):
        """
        :type N: int
        :rtype: int
        """
        # Just try every possible number at each position
#         if N == 0:
#             return 0
        
#         self.count = 0
        
#         def dfs(N, pos, used):
#             if pos == N+1:
#                 self.count += 1
#                 return

#             for i in range(1, N+1):
#                 if used[i] and (i%pos == 0 or pos%i == 0):
#                     used[i] = False
#                     dfs(N, pos+1, used)
#                     used[i] = True
        
#         dfs(N, 1, [True]*(N+1))
#         return self.count

        res = [0]*(N+1)
        def dfs(N, res):
            if N == 1:
                res[0] += 1
                return 

            for i in range(1, len(res)):
                # here we use 0 as marked
                if res[i] == 0 and (N%i == 0 or i%N == 0):
                    res[i] = N
                    dfs(N-1, res)
                    res[i] = 0
        
        dfs(N, res)
        return res[0]
    
    

        # solution 3
        # cache = dict()
        # def solve(idx, nums):
        #     if not nums: return 1
        # # put this combination into cache
        #     key = idx, tuple(nums)
        #     if key in cache: return cache[key]
        #     ans = 0
        #     for i, n in enumerate(nums):
        #         if n % idx == 0 or idx % n == 0:
        #             ans += solve(idx + 1, nums[:i] + nums[i+1:])
        #     cache[key] = ans
        #     return ans
        # return solve(1, range(1, N + 1))
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

211. Add and Search Word - Data structure design

```python
class WordDictionary(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.word_dict = collections.defaultdict(set)
        
    def addWord(self, word):
        """
        Adds a word into the data structure.
        :type word: str
        :rtype: void
        """
        if word:
            self.word_dict[len(word)].add(word)
        
    def search(self, word):
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        :type word: str
        :rtype: bool
        """
        if not word:
            return False
        if '.' not in word:
            return word in self.word_dict[len(word)]
        for v in self.word_dict[len(word)]:
            # match xx.xx.x with yyyyyyy
            for i, ch in enumerate(word):
                if ch != v[i] and ch != '.':
                    break
            else:
                return True
        return False
        


# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)
```

357. Count Numbers with Unique Digits

```python
class Solution(object):
    def countNumbersWithUniqueDigits(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 0: return 1
        if n >= 1 and n <= 10:
            sum = 9
            i = 1
            while(i < n):
                sum *= (10-i)
                i += 1
            return sum + self.countNumbersWithUniqueDigits(n-1)
        if n > 10:
            return self.countNumbersWithUniqueDigits(10)
```

131. Palindrome Partitioning

```python
class Solution(object):
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        def isPalindrome(s):
            for i in range(len(s)/2):
                if s[i] != s[len(s)-1-i]: return False
            return True
    
        def dfs(s, stringlist):
            if len(s) == 0: self.res.append(stringlist)
            for i in range(1, len(s)+1):
                if isPalindrome(s[:i]):
                    dfs(s[i:], stringlist+[s[:i]])
            
        self.res = []
        dfs(s, [])
        return self.res
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

89. Gray Code

```python
class Solution(object):
    def grayCode(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        # solution 1: very smart solution
#         if n == 0:
#             return [0]
        
#         result = self.grayCode(n - 1)
#         seq = list(result)
#         for i in reversed(result):
#             seq.append((1 << (n - 1)) | i)
            
#         return seq
        # solution 2
        results = [0]
        for i in range(n):
            # results += [x + (1 << i) for x in reversed(results)]
            results += [x | (1 << i) for x in reversed(results)]

        return results
    
        # solution 3
        # res = []
        # for i in range(1 << n):
        #     res.append(i ^ (i>>1))
        # return res
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

77. Combinations

```python
class Solution(object):
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        # solution 1 use recursive
        # C(n,k) = sum {C(i,k-1) + [j]} i varies from 0 to n-1 and j varies from 1 to n
        # if k == 0:
        #     return [[]]
        # return [pre + [i] for i in range(1, n+1) for pre in self.combine(i-1, k-1)]

        # solution 2 use backtrack
        # similar to permutation and subsets
        # self.res = []
        # def dfs(temp, index, n, k):
        #     if k == 0:
        #         self.res.append(temp[:])
        #         return
            
        #     for i in xrange(index, n+1):
        #         temp.append(i)
        #         dfs(temp, i+1, n, k-1)
        #         temp.pop()
        
        # dfs([], 1, n, k)
        # return self.res
        
        # solution 3
        # res = []
        # i = 0
        # p =[0]*k
        # while i>=0:
        #     p[i] += 1
        #     if p[i] > n: 
        #         i -= 1
        #     elif i == k-1:
        #         res.append(p[:])
        #     else:
        #         i += 1
        #         p[i] = p[i-1]
        # return res
        
        # solution 4
        from itertools import combinations
        return list(combinations(range(1, n+1), k))
```

60. Permutation Sequence

```python
class Solution(object):
    def getPermutation(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        # same as 46 permutation 
        # solution 1
        
        # self.res = []
        # self.used = [False]*n
        # def dfs(size, string):
        #     if len(string) == size:
        #         self.res.append(string)
            
        #     for i in xrange(n):
    
        #         if self.used[i]:
        #             continue
        #         self.used[i] = True
        #         string += str(i+1)
        #         dfs(n, string)
        #         string = string[:-1]
        #         self.used[i] = False
        # dfs(n, '')
        # return self.res[k-1]
        
        
        # solution 2
        string = '1'
        factorial = [1]
        sum = 1
        # string = '123...n'
        # factorial array 0!,1!...(n-1)!
        for i in xrange(1, n):
            string += str(i+1)
            sum *= i
            factorial.append(sum)
        
        res = ''
        k -= 1
        
        for i in xrange(n):
            index = k/factorial[n-1-i]
            res += string[index]
            # remove that index
            string = string[:index] + string[index+1:]
            k -= index*factorial[n-1-i]
        return res
```