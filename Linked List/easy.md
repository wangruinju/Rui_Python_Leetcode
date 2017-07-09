---
title: "Easy"
author: "Rui Wang"
date: "7/8/2017"
output: html_document
---

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

237. Delete Node in a Linked List

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        # since it does not cover the tail case
        # replace node.val with node.next.val
        # then skip node.next, more like copy node.next to node
        node.val = node.next.val
        node.next = node.next.next
```

83. Remove Duplicates from Sorted List

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
        cur = head
        # cover the edge case
        while cur and cur.next:
            # if cur and cur.next are equal, take two steps
            if cur.val == cur.next.val:
                cur.next = cur.next.next
            # take one step
            else:
                cur = cur.next
                
        return head
```

160. Intersection of Two Linked Lists

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        # save the original nodes
        p1 = headA
        p2 = headB
        
        # check boundary case
        if not p1 or not p2:
            return None
        
        while p1 != p2:
            if p1: p1 = p1.next
            else: p1 = headB
            if p2: p2 = p2.next
            else: p2 = headA
        
        return p1
```

203. Remove Linked List Elements

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        dummy = ListNode(0)
        dummy.next = head
        pre = dummy
        cur = head
        
        while cur != None:
            if cur.val == val:
                pre.next = cur.next
            else:
                pre =  pre.next
            cur = cur.next
                
        return dummy.next
```

206. Reverse Linked List

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # solution 1
        # notice the end of a linked list is None
        # pre = None
        # while head:
        #     cur = head
        #     head = head.next
        #     cur.next = pre
        #     pre = cur
        # return pre
        
        # solution 2 using recursive method
        
        return self.reverse(head)
        
    def reverse(self, node, pre = None):
        # return the last node before None in the linked list
        if node == None:
            return pre
        # keep reverse one by one
        cur = node.next
        node.next = pre
        return self.reverse(cur, node)
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

21. Merge Two Sorted Lists

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        # if none in l1 and l2
        if not l1 or not l2:
            return l1 or l2
        dummy = cur = ListNode(0)
        dummy.next = l1
        
        while l1 and l2:
            if l1.val < l2.val:
                l1 = l1.next
            else:
                cur_temp = cur.next
                l2_temp = l2.next
                cur.next = l2
                l2.next = cur_temp
                l2 = l2_temp
            cur = cur.next
        cur.next = l1 or l2
        return dummy.next
```

