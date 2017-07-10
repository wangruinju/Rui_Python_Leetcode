---
title: "Medium"
author: "Rui Wang"
date: "7/8/2017"
output: html_document
---

150. Evaluate Reverse Polish Notation

```python
class Solution(object):
    def evalRPN(self, tokens):
        """
        :type tokens: List[str]
        :rtype: int
        """
        stack = []
        for token in tokens:
            if token in '+-*/':
                if len(stack) >= 2:
                    temp1 = stack.pop()
                    temp2 = stack.pop()
                    if token == '-':
                        stack.append(temp2-temp1)
                    elif token == '+':
                        stack.append(temp2+temp1)
                    elif token == '*':
                        stack.append(temp2*temp1)
                    else:
                        if temp1*temp2 < 0 and temp2%temp1 != 0:
                            stack.append(temp2/temp1 + 1)
                        else:
                            stack.append(temp2/temp1)
            else:
                stack.append(int(token))
        return stack.pop()
```

503. Next Greater Element II

```python
class Solution(object):
    def nextGreaterElements(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        stack = []
        n = len(nums)
        res = [-1]*n

        for i in range(2*n):
            num = nums[i%n]
            while stack and nums[stack[-1]] < num:
                res[stack.pop()] = num
            if i < n:
                stack.append(i)
        
        return res
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

394. Decode String

```python
class Solution(object):
    def decodeString(self, s):
        """
        :type s: str
        :rtype: str
        """
        # very smart codes
        stack = []
        stack.append(["", 1])
        num = ""
        for ch in s:
            if ch.isdigit():
              num += ch
            elif ch == '[':
                # use a two element array
                stack.append(["", int(num)])
                num = ""
            elif ch == ']':
                st, k = stack.pop()
                # stack[-1][0] += st*k
                stack[-1][0] += ''.join([st]*k)
            else:
                stack[-1][0] += ch
        return stack[0][0]
```

331. Verify Preorder Serialization of a Binary Tree

```python
class Solution(object):
    def isValidSerialization(self, preorder):
        """
        :type preorder: str
        :rtype: bool
        """
        # preorder: root left right
        p = preorder.split(',')
        # notice the number of # is 1 + count of nodes
        # the slot should be 0 at the end otherwise it is not valid
        slot = 1
        for node in p:
            if slot == 0:
                return False
            if node == '#':
                slot -= 1
            else:
                slot += 1
        
        return slot == 0
```

456. 132 Pattern

```python
class Solution(object):
    def find132pattern(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        temp = float('-inf')
        stack = []
        # use the back sequence beacuse 32 in '132' structure are continuous
        # we can always keep a stack flow for 32
        for i in range(len(nums))[::-1]:
            # if any smaller index (1) is less than temp then 132 is true
            if nums[i] < temp:
                return True
            else:
                # the index of temp is larger than i
                while stack and nums[i] > stack[-1]:
                    # temp is 2
                    temp = stack.pop()
            stack.append(nums[i])
        return False
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

173. Binary Search Tree Iterator

```python
# Definition for a  binary tree node
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class BSTIterator(object):
    # solution 1
    # @param root, a binary search tree's root node
    def __init__(self, root):
        self.stack = []
        self.pushLeft(root)

    # @return a boolean, whether we have a next smallest number
    def hasNext(self):            
        # return self.stack
        if self.stack:
            return True
        return False

    # @return an integer, the next smallest number
    def next(self):
        top = self.stack.pop()
        self.pushLeft(top.right)
        return top.val
    
    def pushLeft(self, node):
        while node:
            self.stack.append(node)
            node = node.left

    # solution 2
# @param root, a binary search tree's root node
#     def __init__(self, root):
#         self.tree = []
#         self.inOrderTraverse(root)
#         self.idx = -1
#         self.size = len(self.tree)

#     # @return a boolean, whether we have a next smallest number
#     def hasNext(self):
#         self.idx += 1
#         return self.idx < self.size

#     # @return an integer, the next smallest number
#     def next(self):
#         return self.tree[self.idx]
    
#     def inOrderTraverse(self, root):
#         if root is None:
#             return
#         self.inOrderTraverse(root.left)
#         self.tree.append(root.val)
#         self.inOrderTraverse(root.right)
        

# Your BSTIterator will be called like this:
# i, v = BSTIterator(root), []
# while i.hasNext(): v.append(i.next())
```

341. Flatten Nested List Iterator

```python
# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """
#class NestedInteger(object):
#    def isInteger(self):
#        """
#        @return True if this NestedInteger holds a single integer, rather than a nested list.
#        :rtype bool
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

class NestedIterator(object):
    # Using a stack of [list, index] pairs.
    def __init__(self, nestedList):
        """
        Initialize your data structure here.
        :type nestedList: List[NestedInteger]
        """
        self.stack = [[nestedList, 0]]
        
    # return the next
    def next(self):
        """
        :rtype: int
        """
        self.hasNext()
        nestedList, i = self.stack[-1]
        self.stack[-1][1] += 1
        return nestedList[i].getInteger()
        
    # hasNext is to check whether the next is a single integer
    def hasNext(self):
        """
        :rtype: bool
        """
        s = self.stack
        while s:
            nestedList, i = s[-1]
            if i == len(nestedList):
                s.pop()
            else:
                x = nestedList[i]
                if x.isInteger():
                    return True
                s[-1][1] += 1
                s.append([x.getList(), 0])
        return False

# Your NestedIterator object will be instantiated and called as such:
# i, v = NestedIterator(nestedList), []
# while i.hasNext(): v.append(i.next())
```

144. Binary Tree Preorder Traversal

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):

    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        # solution 1 use dfs
        self.res = []
        def order(root):
            if root:
                self.res.append(root.val)
                if root.left:
                    order(root.left)
                if root.right:
                    order(root.right)
        order(root)
        return self.res
        
        # solution 2 use stack
        # res = []
        # stack = [root]
        
        # while stack:
        #     node = stack.pop()
        #     if node:
        #         res.append(node.val)
        #         stack.append(node.right)
        #         stack.append(node.left)
                
        # return res
```

103. Binary Tree Zigzag Level Order Traversal

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root: return []
        queue = [root]
        res = []
        reverse = False
        while queue:
            n =  len(queue)
            sublist = []
            for i in range(n):
                node = queue.pop()
                if node.left:
                    queue.insert(0, node.left)
                if node.right:
                    queue.insert(0, node.right)
                sublist.append(node.val)
            if not reverse:
                res.append(sublist)
            else:
                res.append(sublist[::-1])
            reverse = not reverse
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

402. Remove K Digits

```python
class Solution(object):
    def removeKdigits(self, num, k):
        """
        :type num: str
        :type k: int
        :rtype: str
        """
        stack = []
        for n in num:
            # in num, if the number in the left is larger than the right than remove it 
            while k and stack and stack[-1] > n:
                stack.pop()
                k -= 1
            stack.append(n)
            
        return ''.join(stack[:-k or None]).lstrip('0') or '0'
```