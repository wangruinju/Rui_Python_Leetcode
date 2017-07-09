---
title: "Easy"
author: "Rui Wang"
date: "7/8/2017"
output: html_document
---

167. Two Sum II - Input array is sorted

```python
class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        ind1 = 0
        ind2 = len(numbers) - 1
        while(numbers[ind1] + numbers[ind2] != target):
            if numbers[ind1] + numbers[ind2] > target:
                ind2 -= 1
            else:
                ind1 += 1
        return [ind1+1, ind2+1]
```

141. Linked List Cycle

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        fast = slow = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if fast == slow:
                return True
        return False
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

26. Remove Duplicates from Sorted Array

```python
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        i = 0
        for num in nums:
            if i < 1 or num > nums[i-1]:
                nums[i] = num
                i += 1
        return i
```

88. Merge Sorted Array

```python
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        while m > 0 and n > 0:
            if nums1[m-1] >= nums2[n-1]:
                nums1[m+n-1] = nums1[m-1]
                m -= 1
            else:
                nums1[m+n-1] = nums2[n-1]
                n -= 1
        if n > 0:
            nums1[:n] = nums2[:n]
```

532. K-diff Pairs in an Array

```python
class Solution(object):
    def findPairs(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        if k > 0:
            return len(set(nums) & set(num + k for num in nums))
        elif k == 0:
            return sum(v > 1 for v in collections.Counter(nums).values())
        else:
            return 0
```

234. Palindrome Linked List

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        # skip the simple case
        if head == None or head.next == None:
            return True
        
        fast = head
        slow = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        # treat the odd case 
        if fast:
            slow = slow.next
        
        # intialize the start, reverse the rest half
        slow = self.reverse(slow)
        fast = head
        
        while slow != None:
            if fast.val != slow.val:
                return False
            fast = fast.next
            slow = slow.next
            
        return True
        
    def reverse(self, head):
        pre = None
        while head:
            cur = head
            head = head.next
            cur.next = pre
            pre = cur
        return pre
```

283. Move Zeroes

```python
class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        j = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[j] = nums[i]
                j += 1
        for i in range(j,len(nums)):
            nums[i] = 0
```

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

27. Remove Element

```python
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        cnt = 0
        for num in nums:
            if num != val:
                nums[cnt] = num
                cnt += 1
        return cnt
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

344. Reverse String

```python
l = list(s)
        # l.reverse()
        n = len(l)
        for i in range(n/2):
            l[i], l[n-1-i] = l[n-1-i], l[i]
        return ''.join(l)
```
