---
title: "Medium"
author: "Rui Wang"
date: "7/8/2017"
output: html_document
---

368. Largest Divisible Subset

```python
class Solution(object):
    def largestDivisibleSubset(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # here is the compact solution
        # S = {-1: set()}
        # for x in sorted(nums):
        #     S[x] = max((S[d] for d in S if x % d == 0), key=len) | {x}
        # return list(max(S.values(), key=len))
        
        # more informative version
        # for each i, get the largest j so i%j == 0 in nums and recorded
        # for print out, just go back from end to start
        n = len(nums)
        count = [None]*n
        pre = [None]*n
        nums.sort()
        d_max = 0
        index = -1
        for i in range(n):
            count[i] = 1
            pre[i] = -1
            for j in range(i)[::-1]:
                if nums[i]%nums[j] == 0:
                    if count[i] < count[j]+1:
                        count[i] = count[j]+1
                        pre[i] = j
            if count[i] > d_max:
                d_max  = count[i]
                index = i
        res = []
        while index != -1:
            res.append(nums[index])
            index =  pre[index]
        
        return res
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

365. Water and Jug Problem

```python
class Solution(object):
    def canMeasureWater(self, x, y, z):
        """
        :type x: int
        :type y: int
        :type z: int
        :rtype: bool
        """
        # The basic idea is to use the property of Bézout's identity and check if z is a multiple of GCD(x, y)
        if x+y < z:
            return False
        if x == z or y ==z or x+y == z:
            return True
        return z%self.gcd(x,y) == 0
        
    def gcd(self, x, y):
        while y:
            temp = y
            y = x%y
            x = temp
        return x
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

343. Integer Break

```python
class Solution(object):
    def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 2: return 1
        if n == 3: return 2
        if n == 4: return 4
        if n%3 == 0: return 3**(n/3)
        if n%3 == 1: return 3**(n/3 - 1)*4
        if n%3 == 2: return 3**(n/3)*2
```

593. Valid Square

```python
class Solution(object):
    def validSquare(self, p1, p2, p3, p4):
        """
        :type p1: List[int]
        :type p2: List[int]
        :type p3: List[int]
        :type p4: List[int]
        :rtype: bool
        """
        res = ([self.d(p1,p2), self.d(p1,p3), self.d(p1,p4), self.d(p2,p3), self.d(p2,p4), self.d(p3,p4)])
        dists = collections.Counter(res)

        return len(dists.values())==2 and 4 in dists.values() and 2 in dists.values()
        
    def d(self, p1, p2):
        return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2
```

166. Fraction to Recurring Decimal

```python
class Solution(object):
    def fractionToDecimal(self, num, den):
        """
        :type numerator: int
        :type denominator: int
        :rtype: str
        """
        # edge case
        if num == 0:
            return "0"
        
        # sign part
        sign = (num>0) ^ (den>0)
        num = abs(num)
        den = abs(den)
        numlist = []
        index = 0
        dict = {}
        # loop part
        loop = ''
        while True:
            numlist.append(str(num/den))
            index += 1
            num = 10*(num%den)
            if num == 0:
                break
            if num in dict:
                loop += ''.join(numlist[dict[num]:index])
                break
            dict[num] = index
            
        # integer part
        ans = numlist[0]
        
        # check whether it has decimals
        if len(numlist) > 1:
            ans += "."
        if loop:
            ans += "".join(numlist[1:len(numlist) - len(loop)]) + "(" + loop + ")"
        else:
            ans += "".join(numlist[1:])
        if sign:
            ans = "-" + ans
        return ans
```

319. Bulb Switcher

```python
class Solution(object):
    def bulbSwitch(self, n):
        """
        :type n: int
        :rtype: int
        """
        # it is another way to calculate how many squares in from 1 to n
        # or that says what the largest i with i*i <= n or int(sqrt(n))
        r = n
        while r*r > n:
            r = (r + n/r) / 2
        return r
```

223. Rectangle Area

```python
class Solution(object):
    def computeArea(self, A, B, C, D, E, F, G, H):
        """
        :type A: int
        :type B: int
        :type C: int
        :type D: int
        :type E: int
        :type F: int
        :type G: int
        :type H: int
        :rtype: int
        """
        return (C-A)*(D-B)+(G-E)*(H-F) - max(min(C,G)-max(A,E),0)*max(min(D,H)-max(B,F),0)
```

592. Fraction Addition and Subtraction

```python
class Solution(object):
    def fractionAddition(self, expression):
        """
        :type expression: str
        :rtype: str
        """
        def gcd(a, b):
            while b:
                temp = b
                b = a%b
                a = temp
            return a

        def lcm(a, b):
            return a * b / gcd(a, b)

        part = ''
        fractions = []
        for c in expression:
            if c in '+-':
                if part: fractions.append(part)
                part = ''
            part += c
        if part: fractions.append(part)

        hi = [int(e.split('/')[0]) for e in fractions]
        lo = [int(e.split('/')[1]) for e in fractions]
        
        LO = reduce(lcm, lo)
        HI = sum(h * LO / l for h, l in zip(hi, lo))
        GCD = abs(gcd(LO, HI))

        return '%s/%s' % (HI / GCD, LO / GCD)
```

279. Perfect Squares

```python
class Solution(object):
    # using dp
    _dp = [0]
    def numSquares(self, n):
        dp = self._dp
        while len(dp) <= n:
            dp += min(dp[-i*i] for i in range(1, int(len(dp)**0.5)+1)) + 1,
        return dp[n]
        
    # another solution is using number theory
    # def numSquares(self, n):
    #     """
    #     :type n: int
    #     :rtype: int
    #     """
    #     while n%4 == 0:
    #         n /= 4
    #     if n%8 == 7:
    #         return 4
    #     temp = self.square(n)
    #     for a in xrange(temp+1):
    #         b = self.square(n-a*a)
    #         if a*a + b*b == n:
    #             return int(a!=0) + int(b!=0)
    #     return 3

    # def square(self, n):
    #     r = n
    #     while r*r > n:
    #         r = (r+n/r)/2
    #     return r
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

264. Ugly Number II

```python
class Solution(object):
    def nthUglyNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        p1, p2, p3 = 0, 0, 0 #pointers in the following list
        
        q = [0] * n
        q[0] = 1
        
        for i in range(1, n):
            t1, t2, t3 = q[p1] * 2, q[p2] * 3, q[p3] * 5
            q[i] = min(t1, t2, t3)
            if q[i] == t1: p1 += 1
            if q[i] == t2: p2 += 1
            if q[i] == t3: p3 += 1
            
        return q[-1]
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

535. Encode and Decode TinyURL

```python
class Codec:
    # def __init__(self):
    #     self.urls = []

    # def encode(self, longUrl):
    #     """Encodes a URL to a shortened URL.
        
    #     :type longUrl: str
    #     :rtype: str
    #     """
    #     self.urls.append(longUrl)
    #     return 'http://tinyurl.com/' + str(len(self.urls) - 1)
        
    # def decode(self, shortUrl):
    #     """Decodes a shortened URL to its original URL.
        
    #     :type shortUrl: str
    #     :rtype: str
    #     """
    #     return self.urls[int(shortUrl.split('/')[-1])]
        
    # or we can use two hash table and map each other
    alphabet = string.ascii_letters + '0123456789'

    def __init__(self):
        self.url2code = {}
        self.code2url = {}

    def encode(self, longUrl):
        while longUrl not in self.url2code:
            code = ''.join(random.choice(Codec.alphabet) for _ in range(6))
            if code not in self.code2url:
                self.code2url[code] = longUrl
                self.url2code[longUrl] = code
        return 'http://tinyurl.com/' + self.url2code[longUrl]

    def decode(self, shortUrl):
        return self.code2url[shortUrl[-6:]]

        

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.decode(codec.encode(url))
```

2. Add Two Numbers

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        dummy = cur = ListNode(0)
        # mark the add digit
        comb = 0
        
        while l1 or l2:
            # set comb to val from last round
            val = comb
            if l1:
                val += l1.val
                l1 = l1.next
            if l2:
                val += l2.val
                l2 = l2.next
            comb = val/10
            val = val%10
            cur.next = ListNode(val)
            cur = cur.next
        # if the last round added up larger than 10, then leave 1 in the end
        if comb == 1:
            cur.next = ListNode(1)
        return dummy.next
```

313. Super Ugly Number

```python
class Solution(object):
    def nthSuperUglyNumber(self, n, primes):
        """
        :type n: int
        :type primes: List[int]
        :rtype: int
        """
        res = [0]*n
        res[0] = 1
        p = len(primes)
        index = [0]*p
        for i in xrange(1,n):
            res[i] = float('inf')
            for j in xrange(p):
                res[i] = min(res[i], primes[j]*res[index[j]])
            for j in xrange(p):
                if primes[j]*res[index[j]] == res[i]:
                    index[j] += 1
        return res[-1]
        
        # uglies = [1]
        # merged = heapq.merge(*map(lambda p: (u*p for u in uglies), primes))
        # uniqed = (u for u, _ in itertools.groupby(merged))
        # map(uglies.append, itertools.islice(uniqed, n-1))
        # return uglies[-1]
```

523. Continuous Subarray Sum

```python
class Solution(object):
    def checkSubarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        dmap = {0 : -1}
        total = 0
        for i, n in enumerate(nums):
            total += n
            m = total % k if k else total
            # record this index if it is not in dmap
            if m not in dmap: dmap[m] = i
            # if m in dmap and this index is larger than the previous index
            # which ensure this subsum is a multiple of k
            elif dmap[m] + 1 < i: return True
        return False
```

29. Divide Two Integers

```python
class Solution(object):
    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """
        sign = (dividend < 0) ^ (divisor < 0)
        dividend, divisor = abs(dividend), abs(divisor)
        res = 0
        while dividend >= divisor:
            temp, i = divisor, 1
            while dividend >= temp:
                dividend -= temp
                res += i
                i <<= 1
                temp <<= 1
        if sign:
            res = -res
        return min(max(-2147483648, res), 2147483647)
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

462. Minimum Moves to Equal Array Elements II

```python
class Solution(object):
    def minMoves2(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        medium = sorted(nums)[len(nums)/2]
        return sum(abs(num -medium) for num in nums)
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

50. Pow(x, n)

```python
class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n==0:
            return 1
        elif n<0:
            return 1/self.myPow(x, -n)
        else:
            half = self.myPow(x,n/2)
            if (n%2) == 0:
                return half*half
            else:
                return half*half*x
```

423. Reconstruct Original Digits from English

```python
class Solution(object):
    def originalDigits(self, s):
        """
        :type s: str
        :rtype: str
        """
        char = [0]*26
        for string in s:
            char[ord(string) - ord('a')] += 1
        
        cnt = [0]*10
        # 2: "w"
        cnt[2] = char[22]
        # 6: "x"
        cnt[6] = char[23]
        # 7: "s" - "x"
        cnt[7] = char[18] - cnt[6]
        # 5: "v"
        cnt[5] = char[21] - cnt[7]
        # 8: "g"
        cnt[8] = char[6]
        # 3: "h" - "g"
        cnt[3] = char[7] - cnt[8]
        # 0: "z"
        cnt[0] = char[25]
        # 4: "f" - "v"
        cnt[4] = char[5] - cnt[5]
        # 1: "o" -  0 - 4
        cnt[1] = char[14] - cnt[0] - cnt[2] - cnt[4] 
        # 9: "i" - 6 -8
        cnt[9] = char[8] - cnt[5] - cnt[6] - cnt[8]
        
        return ''.join([str(i)*cnt[i] for i in xrange(10)])
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

413. Arithmetic Slices

```python
class Solution(object):
    def numberOfArithmeticSlices(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        # my original solution, very bad but pass
        # cnt = []
        # ans = 0
        # for i in xrange(len(A)-2):
        #     if A[i] + A[i+2] == A[i+1]*2:
        #         cnt.append(1)
        #     else:
        #         cnt.append(0)
        # print cnt
        # while cnt:
        #     ans += sum(cnt)
        #     for i in xrange(len(cnt)-1):
        #         cnt[i] = cnt[i]&cnt[i+1]
        #     cnt.pop()
        # return ans
        
        # best solution put all in one loop
        cur = 0
        sum = 0
        for i in xrange(2, len(A)):
            if A[i] - A[i-1] == A[i-1] - A[i-2]:
            # 0+1+2+... if continous
                cur += 1
                sum += cur
            # discontinous then reset cur = 0
            else:
                cur = 0
        return sum
```

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

396. Rotate Function

```python
class Solution(object):
    def maxRotateFunction(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        n = len(A)
        if n == 0 or n == 1 or n>10**5:
            return 0
        
        # define the original sum
        s = 0
        for i in range(n):
            s += i*A[i]
       
        t = sum(A)
        news = s
        for j in range(1,n):
            s += t - n * A[-j]
            news = max(s, news)
        return news
```

372. Super Pow

```python
class Solution(object):
    def superPow(self, a, b):
        """
        :type a: int
        :type b: List[int]
        :rtype: int
        """
        # solution 1: use pow function
        # ans, pow = 1, a
        # for n in b[::-1]:
        #     ans = (ans * (pow ** n) % 1337) % 1337
        #     pow = (pow ** 10) % 1337
        # return ans
        
        # solution 2 optimize with using pow function
        
    #     ans, pow = 1, a
    #     for n in b[::-1]:
    #         ans = (ans * self.quickPow(pow, n, 1337)) % 1337
    #         pow = self.quickPow(pow, 10, 1337)
    #     return ans

    # def quickPow(self, a, b, m):
    #     ans = 1
    #     while b > 0:
    #         # binary search
    #         # if b is odd, ans multiple by one a
    #         if b & 1: ans *= a
    #         # get the residual from a^2 % m
    #         a = (a * a) % m
    #         # b = b/2
    #         b >>= 1
    #     return ans
        
        # solution 3 using Euler's theorem and Fermat's little theorem:
        return 0 if a % 1337 == 0 else pow(a, reduce(lambda x, y: (x * 10 + y) % 1140, b) + 1140, 1337)
```