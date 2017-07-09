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

9. Palindrome Number

```python
class Solution(object):
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x < 0:
            return False
        new_x = int(str(x)[::-1])
        return new_x == x
```

598. Range Addition II

```python
class Solution(object):
    def maxCount(self, m, n, ops):
        """
        :type m: int
        :type n: int
        :type ops: List[List[int]]
        :rtype: int
        """
        matrix_rmin = m
        matrix_cmin = n
        for op in ops:
            matrix_rmin = min(matrix_rmin, op[0])
            matrix_cmin = min(matrix_cmin, op[1])
            
        return matrix_rmin*matrix_cmin
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

507. Perfect Number

```python
class Solution(object):
    def checkPerfectNumber(self, num):
        """
        :type num: int
        :rtype: bool
        """
        if num <= 0: return False
        sum = 0
        sqrt = int(num**0.5)
        for i in range(1,sqrt+1):
            if num%i == 0:
                sum += i + num/i
        if num == sqrt**2: sum -= sqrt        
        return sum == 2*num
```

453. Minimum Moves to Equal Array Elements

```python
class Solution(object):
    def minMoves(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return sum(nums) - min(nums)*len(nums)
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

415. Add Strings

```python
class Solution(object):
    def addStrings(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        flag = 0
        addstr = ''
        t1 = len(num1)
        t2 = len(num2)
        while t1 or t2 or flag:
            digit = flag
            if t1:
                t1 -= 1
                digit += ord(num1[t1]) - 48
                
            if t2:
                t2 -= 1
                digit += ord(num2[t2]) - 48
            flag = digit > 9
            addstr += str(digit%10)
            
        return addstr[::-1]
```

258. Add Digits

```python
class Solution(object):
    def addDigits(self, num):
        """
        :type num: int
        :rtype: int
        """
        if num == 0:
            return 0
        elif num%9 == 0:
            return 9
        else:
            return num%9
```

263. Ugly Number

```python
class Solution(object):
    def isUgly(self, num):
        """
        :type num: int
        :rtype: bool
        """
        if num > 0:
            for i in 2, 3, 5:
                while num%i == 0:
                    num /= i
        return num > 0 and num == 1
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

7. Reverse Integer

```python
class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        # result = int(str(x)[::-1]) if x >= 0 else int('-'+str(x)[1:][::-1])
        # if result <= 2147483647 and result >= -2147483648:
        #     return result 
        # else:
        #     return 0
        
        res = 0
        sign = 1 if x >= 0 else -1
        x *= sign
        
        while x:
            res = res*10 + x%10
            x /= 10
            if res > 2147483647:
                return 0
        return res*sign
```

633. Sum of Square Numbers

```python
class Solution(object):
    def judgeSquareSum(self, c):
        """
        :type c: int
        :rtype: bool
        """
        if c < 0:
            return False
    
        left = 0
        right = int(c**0.5)
        while left <= right:
            temp = left*left + right*right
            if temp == c:
                return True
            elif temp > c:
                right -= 1
            else:
                left += 1
        return False
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

172. Factorial Trailing Zeroes

```python
class Solution(object):
    def trailingZeroes(self, n):
        """
        :type n: int
        :rtype: int
        """
        count = 0
        if n<=4:
            return count
        while n >= 5:
            count += n/5
            n = n/5
        return count
```

171. Excel Sheet Column Number

```python
class Solution(object):
    def titleToNumber(self, s):
        """
        :type s: str
        :rtype: int
        """
        str_list =  list(s)
        str_list.reverse()
        num = 0
        for i in range(len(str_list)):
            num += (ord(str_list[i]) - 64)*(26**i)
        return num
```

168. Excel Sheet Column Title

```python
class Solution(object):
    def convertToTitle(self, n):
        """
        :type n: int
        :rtype: str
        """
        s = ''
        while n > 0:
            s += chr((n-1)%26+65)
            n = (n-1)/26
        return s[::-1]
```

326. Power of Three

```python
class Solution(object):
    def isPowerOfThree(self, n):
        """
        :type n: int
        :rtype: bool
        """
        if n<0:
            return False
        elif n==0:
            return False
        elif n==1:
            return True
        elif n%3!=0:
            return False
        else:
            return self.isPowerOfThree(n/3)
```

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

67. Add Binary

```python
class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        return bin(int(a,2) + int(b,2))[2:]
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

400. Nth Digit

```python
class Solution(object):
    def findNthDigit(self, n):
        """
        :type n: int
        :rtype: int
        """
        i = 1
        while n>9*10**(i-1)*i:
            n -= 9*10**(i-1)*i
            i += 1
        s = (n-1)/i
        t = (n-1)%i
        return int(str(10**(i-1)+s)[t])
```