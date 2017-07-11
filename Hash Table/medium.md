---
title: "Medium"
author: "Rui Wang"
date: "7/8/2017"
output: html_document
---

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

299. Bulls and Cows

```python
class Solution(object):
    def getHint(self, secret, guess):
        """
        :type secret: str
        :type guess: str
        :rtype: str
        """

        bull_cnt = 0
        cow_cnt = 0
        
        # num1 = [0]*10
        # num2 = [0]*10
        
        # for i in xrange(len(secret)):
        #     if secret[i] == guess[i]:
        #         bull_cnt += 1
        #     else:
        #         num1[ord(secret[i]) - ord('0')] += 1
        #         num2[ord(guess[i]) - ord('0')] += 1
        
        # for i in xrange(10):
        #     cow_cnt += min(num1[i], num2[i])
        
        num = [0]*10
        for i in xrange(len(secret)):
            s = ord(secret[i]) -ord('0')
            g = ord(guess[i]) -ord('0')
            if s == g:
                bull_cnt += 1
            else:
                if num[s] < 0:
                    cow_cnt += 1
                if num[g] > 0:
                    cow_cnt += 1
                num[s] += 1
                num[g] -= 1
            
        return str(bull_cnt) + 'A' + str(cow_cnt) + 'B'
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

274. H-Index

```python
class Solution(object):
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """
        if not citations:
            return 0
        n = len(citations)
        bucket = [0]*(n+1)
        
        for c in citations:
            if c >= n:
                bucket[n] += 1
            else:
                bucket[c] += 1
                
        cnt = 0
        for i in range(n+1)[::-1]:
            cnt += bucket[i]
            if cnt >= i:
                return i
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

454. 4Sum II

```python
class Solution(object):
    def fourSumCount(self, A, B, C, D):
        """
        :type A: List[int]
        :type B: List[int]
        :type C: List[int]
        :type D: List[int]
        :rtype: int
        """
        ans = 0
        cnt = {}
        for a in A:
            for b in B:
                cnt[a + b] = cnt.get(a + b, 0) + 1
        for c in C:
            for d in D:
                ans += cnt.get(-(c + d), 0)
        return ans
```

451. Sort Characters By Frequency

```python
class Solution(object):
    def frequencySort(self, s):
        """
        :type s: str
        :rtype: str
        """
        # return ''.join(c * t for c, t in collections.Counter(s).most_common())
        
        char_cnt = {}
        for char in s:
            char_cnt[char] = char_cnt.get(char, 0) + 1
        cnt_char = {}
        for key in char_cnt.keys():
            cnt_char[char_cnt[key]] = cnt_char.get(char_cnt[key], '') + key
        
        ans = ''
        # if sort is not allowed
        # start from len(s) and decrease to 0
        for i in sorted(cnt_char.keys(), reverse = True):
            for char in cnt_char[i]:
                ans += char*i
        return ans
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

18. 4Sum

```python
class Solution(object):
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        n = len(nums)
        res = []
        if n < 4: 
            return res
        nums.sort()
        
        for i in range(n-3):
            if i > 0 and nums[i]==nums[i-1]:
                continue
            if nums[i]+nums[i+1]+nums[i+2]+nums[i+3]>target: 
                break
            if nums[i]+nums[n-3]+nums[n-2]+nums[n-1]<target:
                continue
        
            for j in range(i+1, n-2):
                if j > i+1 and nums[j]==nums[j-1]:
                    continue
                if nums[i]+nums[j]+nums[j+1]+nums[j+2]>target:
                    break
                if nums[i]+nums[j]+nums[n-2]+nums[n-1]<target:
                    continue
                
                left = j+1; right = n-1
                while(left < right):
                    temp = nums[i]+nums[j]+nums[left]+nums[right]
                    if temp < target:
                        left += 1
                    elif temp > target:
                        right -= 1
                    else:
                        res.append([nums[i], nums[j], nums[left], nums[right]])
                        while(left < right and nums[left]==nums[left+1]):
                            left += 1
                        while(left < right and nums[right]==nums[right-1]):
                            right -= 1
                        left += 1; right -= 1
                    
        return res
```

94. Binary Tree Inorder Traversal

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        self.res = []
        if not root:
            return []
        # inorder: left root right
        def dfs(root):
            
            if not root:
                return
            if root.left:
                dfs(root.left)
                
            self.res.append(root.val)
            
            if root.right:
                dfs(root.right)
        dfs(root)
        return self.res
```

380. Insert Delete GetRandom O(1)

```python
class RandomizedSet(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.datamap = {}
        self.datalist = []

    def insert(self, val):
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        if val in self.datamap:
            return False
        else:
            self.datamap[val] = len(self.datalist)
            self.datalist.append(val)
            return True
        

    def remove(self, val):
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        :type val: int
        :rtype: bool
        """
        if val not in self.datamap:
            return False
        else:
            index = self.datamap[val]
            tail = self.datalist.pop()
            # if the val is not the tail of datalist
            # if the tail corresponding to the index, we have already removed it
            if index < len(self.datalist):
                self.datalist[index] = tail
                self.datamap[tail] = index
            del self.datamap[val]
            
            return True
        


    def getRandom(self):
        """
        Get a random element from the set.
        :rtype: int
        """
        return random.choice(self.datalist)
        


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()
```

138. Copy List with Random Pointer

```python
# Definition for singly-linked list with a random pointer.
# class RandomListNode(object):
#     def __init__(self, x):
#         self.label = x
#         self.next = None
#         self.random = None

class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: RandomListNode
        :rtype: RandomListNode
        """
        dic = {}
        m = n = head
        
        while m:
            dic[m] = RandomListNode(m.label)
            m = m.next
        
        while n:
            dic[n].next = dic.get(n.next)
            dic[n].random = dic.get(n.random)
            n = n.next
        return dic.get(head)
```

554. Brick Wall

```python
class Solution(object):
    def leastBricks(self, wall):
        """
        :type wall: List[List[int]]
        :rtype: int
        """
        dict = {}
        count = 0

        for list in wall:
            listsum = 0
            for i in xrange(len(list)-1):
                listsum += list[i]
                dict[listsum] = dict.get(listsum, 0) + 1
                count = max(count, dict[listsum])
        return len(wall) - count
```

355. Design Twitter

```python
class Twitter(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.timer = itertools.count(step=-1)
        self.tweets = collections.defaultdict(collections.deque)
        self.followees = collections.defaultdict(set)
        

    def postTweet(self, userId, tweetId):
        """
        Compose a new tweet.
        :type userId: int
        :type tweetId: int
        :rtype: void
        """
        self.tweets[userId].appendleft((next(self.timer), tweetId))


    def getNewsFeed(self, userId):
        """
        Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
        :type userId: int
        :rtype: List[int]
        """
        tweets = heapq.merge(*(self.tweets[u] for u in self.followees[userId] | {userId}))
        return [t for _, t in itertools.islice(tweets, 10)]

    def follow(self, followerId, followeeId):
        """
        Follower follows a followee. If the operation is invalid, it should be a no-op.
        :type followerId: int
        :type followeeId: int
        :rtype: void
        """
        self.followees[followerId].add(followeeId)
        
        

    def unfollow(self, followerId, followeeId):
        """
        Follower unfollows a followee. If the operation is invalid, it should be a no-op.
        :type followerId: int
        :type followeeId: int
        :rtype: void
        """
        self.followees[followerId].discard(followeeId)



# Your Twitter object will be instantiated and called as such:
# obj = Twitter()
# obj.postTweet(userId,tweetId)
# param_2 = obj.getNewsFeed(userId)
# obj.follow(followerId,followeeId)
# obj.unfollow(followerId,followeeId)
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