---
title: "Medium"
author: "Rui Wang"
date: "7/8/2017"
output: html_document
---

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

148. Sort List

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # the basic idea is mergesort
        # skip the boundary case
        if not head or not head.next:
            return head
        # step 2, sort each small list
        slow = self.findmid(head)
        l1 = self.sortList(head)
        l2 = self.sortList(slow)
        return self.mergelist(l1, l2)
        
        
    def findmid(self, head):
        # step 1, split the list into two list
        # initial pre node
        pre = None
        slow = fast = head
        while fast and fast.next:
            pre = slow
            slow = slow.next
            fast = fast.next.next
            
        # break it into half
        pre.next = None
        return slow
        
        
        
    
    def mergelist(self, l1, l2):
        dummy = cur = ListNode(0)
        while l1 and l2:
            if l1.val < l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next
            
        if l1:
            cur.next = l1
        if l2:
            cur.next = l2
            
        return dummy.next
```

147. Insertion Sort List

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def insertionSortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummy = ListNode(0)
        while head:
            pre = dummy
            while pre.next and pre.next.val < head.val:
                pre = pre.next
            temp = head.next
            head.next = pre.next
            pre.next = head
            head = temp
        return dummy.next
```

143. Reorder List

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: void Do not return anything, modify head in-place instead.
        """
        if not head:
            return
        mid = self.findmid(head)
        tail = self.reverse(mid)
        self.mergelist(head, tail)
        
    # find the middle in a linked list  
    def findmid(self, head):
        fast = slow = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        # check the 1,2,3,4 example, the result is 1,4,2,3 not 1,4,3,2
        # so if n is even or odd, it break into n/2 + 1, n/2 -1 (front half = back half + 2)
        mid = slow.next
        # this is the pivot
        # set the slow.next to None and break it
        slow.next = None
        
        return mid
        
    # merge two list one by one   
    def mergelist(self, head, tail):
        cur1, cur2 = head, tail
        while cur2:
            temp1, temp2 = cur1.next, cur2.next
            cur1.next = cur2
            cur2.next = temp1
            cur1 = temp1
            cur2 = temp2
            
            
        return head
        
    # reverse a list
    def reverse(self, head):
        pre = None
        while head:
            cur = head
            head = head.next
            cur.next = pre
            pre = cur
        return pre
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

109. Convert Sorted List to Binary Search Tree

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sortedListToBST(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """
        if not head:
            return None
        return self.tobst(head, None)

    def tobst(self, head, tail):
        if head == tail: return None
    
        fast = slow = head
        while fast != tail and fast.next != tail:
            slow = slow.next
            fast = fast.next.next

        node = TreeNode(slow.val)
        node.left = self.tobst(head, slow)
        node.right = self.tobst(slow.next, tail)

        return node
```

92. Reverse Linked List II

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """
        if not head or not head.next:
            return head
        
        dummy = cur = ListNode(0)
        dummy.next = head
        
        # find the node before mNode
        for i in range(1,m):
            cur = cur.next
        
        pre = cur
        # the mth node
        mNode = cur.next
        nNode = cur.next
        pos = nNode.next
        # reverse the node from m to n
        # pre - pos change to pos - pre
        for i in range(m,n):
            temp = pos.next
            pos.next = nNode
            nNode = pos
            pos = temp
        
        # dummy - mNode - pos - pre - nNode - None
        mNode.next = pos;
        pre.next = nNode;
        
        return dummy.next
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

82. Remove Duplicates from Sorted List II

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            return head
        
        dummy = cur = ListNode(0)
        dummy.next = head
        
        while cur.next and cur.next.next:
            # if duplicate, take two steps
            if cur.next.val == cur.next.next.val:
                val = cur.next.val
                while cur.next and cur.next.val == val:
                    cur.next = cur.next.next
            
            # take one step
            else:
                cur = cur.next
        
        return dummy.next
```

445. Add Two Numbers II

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
        
        l1 = self.reverse(l1)
        l2 = self.reverse(l2)
        
        dummy = cur = ListNode(0)
        comb = 0
        
        while l1 or l2:
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
        if comb == 1:
            cur.next = ListNode(1)

        return self.reverse(dummy.next)
            
        
    def reverse(self, node, pre = None):
        # return the last node before None in the linked list
        if node == None:
            return pre
        # keep reverse one by one
        cur = node.next
        node.next = pre
        return self.reverse(cur, node)
```

328. Odd Even Linked List

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head and head.next:
            odd = head
            even = head.next
            dummy = even
            
            while even and even.next:
                odd.next = odd.next.next
                even.next = even.next.next
                odd = odd.next
                even = even.next
            odd.next = dummy
        return head
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

24. Swap Nodes in Pairs

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # solution 1
        # if not head:
        #     return head
            
        # dummy = ListNode(0)
        # dummy.next = head
        # pre = dummy
        
        # while pre.next and pre.next.next:
            
        #     p1 = pre.next
        #     p2 = pre.next.next
            
        #     pre.next = p2
        #     p1.next = p2.next
        #     p2.next = p1
            
        #     pre = pre.next.next
        
        # return dummy.next
        
        # solution 2
        # use recursion
        
        if not head or not head.next:
            return head
        pos = head.next
        head.next = self.swapPairs(head.next.next)
        pos.next = head
        return pos
```