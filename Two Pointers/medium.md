---
title: "Medium"
author: "Rui Wang"
date: "7/8/2017"
output: html_document
---

209. Minimum Size Subarray Sum

```python
class Solution(object):
    def minSubArrayLen(self, s, nums):
        """
        :type s: int
        :type nums: List[int]
        :rtype: int
        """
        # solution 1 using O(n)
        # if not nums:
        #     return 0
        # left = 0
        # right = 0
        # sum = 0
        # res = len(nums)+1
        # while right < len(nums):
        #     while sum < s and right < len(nums):
        #         sum += nums[right]
        #         right += 1
        #     while sum >= s:
        #         res = min(res, right-left)
        #         sum -= nums[left]
        #         left += 1
        
        # return res if res != len(nums)+1 else 0
        
        res = len(nums)+1
        left = 0
        sum = 0
        for i in xrange(len(nums)):
            sum += nums[i]
            while left <= i and sum >= s:
                res = min(res, i-left+1)
                sum -= nums[left]
                left += 1
        return res if res != len(nums)+1 else 0
        
        # solution 2 using binary search
    #     n = len(nums)
    #     sums = [0]*(n+1)
    #     for i in xrange(1, n+1):
    #         sums[i] = sums[i-1] + nums[i-1]
        
    #     minLen = n+1
    #     for i in xrange(n+1):
    #         # sums[j] satisfy sums[j] >= sums[i] + s
    #         # the length is j - i
    #         end = self.helper(i+1, n, sums[i]+s, sums)
    #         # end is out of the arrays, no subarray qualify
    #         if end == n+1: break
    #         minLen = min(minLen, end - i)
    #     return minLen if minLen != n+1 else 0
    
    # def helper(self, lo, hi, key, sums):
    #     while lo <= hi:
    #         mid = (hi + lo)/2
    #         if sums[mid] >= key:
    #             hi = mid - 1
    #         else:
    #             lo = mid + 1
    #     return lo
```

61. Rotate List

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def rotateRight(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        if not head or not head.next:
            return head
        
        tail = head
        cnt = 1
        while tail.next:
            cnt += 1
            tail = tail.next
        
        # cycle the list
        tail.next = head
        
        k %= cnt
        # if k == 0, stay still
        if k:
            for i in range(cnt-k):
                tail = tail.next
        # find the cut point
        new = tail.next
        tail.next = None
        return new
```

524. Longest Word in Dictionary through Deleting

```python
class Solution(object):
    def findLongestWord(self, s, d):
        """
        :type s: str
        :type d: List[str]
        :rtype: str
        """
        
        # res = []
        # s_max = 0
        # for string in d:
        #     i = 0
        #     j = 0
            
        #     while i < len(s) and j < len(string):
        #         if s[i] == string[j]:
        #             i += 1
        #             j += 1
        #         else:
        #             i += 1
        #         if j == len(string):
        #             res.append(string)
        #             s_max = max(s_max, len(string))
        # res.sort()
        # for string in res:
        #     if len(string) == s_max:
        #         return string
        # return ''
        
        best = ''
        for x in d:
            # first compare the length and second compare the lexicographical order
            if (-len(x), x) < (-len(best), best):
                it = iter(s)
                if all(c in it for c in x):
                    best = x
        return best
```

142. Linked List Cycle II

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # check the cycle
        # find slow and fast intersects
        # reset slow or fast from start point at the same pace
        fast = slow = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if fast == slow:
                break
        else:
            return None
        
        while head != slow:
            head = head.next
            slow = slow.next
        return head
```

567. Permutation in String

```python
class Solution(object):
    def checkInclusion(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        # solution 1 using hash table
        # use two pointers in one counting hash table
        # l1, l2 = len(s1), len(s2)
        # c1 = collections.Counter(s1)
        # c2 = collections.Counter()
        # p = q = 0
        # while q < l2:
        #     c2[s2[q]] += 1
        #     if c1 == c2:
        #         return True
        #     q += 1
        #     if q - p + 1 > l1:
        #         c2[s2[p]] -= 1
        #         if c2[s2[p]] == 0:
        #             del c2[s2[p]]
        #         p += 1
        # return False
        
        # solution 2 using sliding window
        # 在比较s2的窗口部分与s1的字符个数时，跟踪s2的窗口边界发生变化的字符
        # 通过计数器cnt统计s2的窗口部分与s1相等的字符个数,
        # 当cnt == len(set(s1))时，返回True
        # l1, l2 = len(s1), len(s2)
        # c1 = collections.Counter(s1)
        # c2 = collections.Counter()
        # cnt = 0
        # p = q = 0
        # while q < l2:
        #     c2[s2[q]] += 1
        #     if c1[s2[q]] == c2[s2[q]]:
        #         cnt += 1
        #     if cnt == len(c1):
        #         return True
        #     q += 1
        #     if q - p + 1 > l1:
        #         if c1[s2[p]] == c2[s2[p]]:
        #             cnt -= 1
        #         c2[s2[p]] -= 1
        #         if c2[s2[p]] == 0:
        #             del c2[s2[p]]
        #         p += 1
        # return False
        
        # solution 3 change hash table to array [0]*26; the comparison can speed up

        l1 = [0]*26
        l2 = [0]*26
        for x in s1:
            l1[ord(x) - ord('a')] += 1
        
        for i in xrange(len(s2)):
            l2[ord(s2[i]) - ord('a')] += 1
            if i >= len(s1):
                l2[ord(s2[i-len(s1)]) - ord('a')] -= 1
            if l1 == l2:
                return True
        return False
```

75. Sort Colors

```python
class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        left = 0
        right = len(nums) - 1
        for i in range(len(nums)):
            while(nums[i] == 2 and i < right):
                temp = nums[i]
                nums[i] = nums[right]
                nums[right] = temp
                right -= 1
            while(nums[i] == 0 and i > left):
                temp = nums[i]
                nums[i] = nums[left]
                nums[left] = temp
                left += 1
```

86. Partition List

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def partition(self, head, x):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        dummy1 = cur1 = ListNode(0)
        dummy2 = cur2 = ListNode(0)
        while head:
            if head.val < x:
                cur1.next = head
                cur1 = cur1.next
            else:
                cur2.next = head
                cur2 = cur2.next
            head = head.next
            
        cur2.next = None
        cur1.next = dummy2.next
        return dummy1.next
```

80. Remove Duplicates from Sorted Array II

```python
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        i = 0
        for num in nums:
            # skip the first k = 2 elements, move if num > nums[i-k]
            if i < 2 or num > nums[i-2]:
                nums[i] = num
                i += 1
        return i
```

15. 3Sum

```python
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        n = len(nums)
        res = []
        for i in range(n-2):
            # get a unique start
            if i > 0 and nums[i] == nums[i-1]:
                continue
            left = i+1; right = n-1
            while left < right:
                target = nums[i] + nums[left] + nums[right]
                if target < 0:
                    left += 1
                elif target > 0:
                    right -= 1
                else:
                    # if matched, attach the result
                    res.append([nums[i], nums[left], nums[right]])
                    # remove the duplicate
                    while left < right and nums[left] == nums[left+1]:
                        left += 1
                    while left < right and nums[right] == nums[right-1]:
                        right -= 1
                    # next search
                    left += 1; right -= 1
        return res
```

16. 3Sum Closest

```python
class Solution(object):
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        n = len(nums)
        nums.sort()
        # initialize res with the first three numbers
        res = nums[0] + nums[1] + nums[2]
        for i in range(n-2):
            left = i+1; right = n-1
            while left < right:
                sum = nums[i] + nums[left] + nums[right]
                if sum == target:
                    return sum
                if abs(sum - target) < abs(res - target):
                    res = sum
                if sum < target:
                    left += 1
                elif sum > target:
                    right -= 1
        return res
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

19. Remove Nth Node From End of List

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        
        dummy = ListNode(0)
        dummy.next = head
        slow = fast = dummy
        
        for i in range(n+1):
            fast = fast.next
        
        while fast:
            fast = fast.next
            slow = slow.next
        
        slow.next = slow.next.next
        return dummy.next
```

287. Find the Duplicate Number

```python
class Solution(object):
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        slow = 0
        fast = 0
        while(True):
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break
        fast = 0
        while(True):
            slow = nums[slow]
            fast = nums[fast]
            if slow == fast:
                return slow
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

11. Container With Most Water

```python
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        left = 0
        right = len(height) - 1
        # intial the water as 0
        res = 0
        while (left < right):
            h = min(height[left], height[right])
            res = max(res, h*(right-left))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return res
```