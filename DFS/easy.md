---
title: "Easy"
author: "Rui Wang"
date: "7/8/2017"
output: html_document
---

100. Same Tree

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        if p == None and q == None: return True
        elif p == None or q == None: return False
        else:
            return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
```

101. Symmetric Tree

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if root == None: return True
        return self.ismirrortree(root.left, root.right)

    def ismirrortree(self, p, q):
        if not p or not q:
            return p == q
        else:
            return p.val == q.val and self.ismirrortree(p.left, q.right) and self.ismirrortree(p.right, q.left)
```

104. Maximum Depth of Binary Tree

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
```

257. Binary Tree Paths

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        res = []
        if not root:
            return res
        self.addpath(root, "", res)
        return res
        
    def addpath(self, root, path, res):
        if not root.left and not root.right:
            res.append(path + str(root.val))
        if root.left:
            self.addpath(root.left, path + str(root.val) +"->", res)
        if root.right:
            self.addpath(root.right, path + str(root.val) +"->", res)
```

112. Path Sum

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        if root == None:
            return False
        
        if root.left == None and root.right == None and sum == root.val:
            return True
        
        return self.hasPathSum(root.left, sum - root.val) or self.hasPathSum(root.right, sum - root.val)
```

108. Convert Sorted Array to Binary Search Tree

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if len(nums) == 0:
            return None
            
        mid = len(nums) >> 1
        node = TreeNode(nums[mid])
        node.left = self.sortedArrayToBST(nums[:mid])
        node.right = self.sortedArrayToBST(nums[mid+1:])
        return node
```

111. Minimum Depth of Binary Tree

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        elif not root.left or not root.right:
            return max(self.minDepth(root.left), self.minDepth(root.right)) + 1
        else:
            return min(self.minDepth(root.left), self.minDepth(root.right)) + 1
```

110. Balanced Binary Tree

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
    # solution 1: top down
    #     if root == None: return True
    #     left = self.depth(root.left)
    #     right = self.depth(root.right)
    #     return abs(left-right) <= 1 and self.isBalanced(root.left) and self.isBalanced(root.right)
        
    # def depth(self, root):
    #     if root == None:
    #         return 0
    #     return max(self.depth(root.left), self.depth(root.right)) + 1
    
    # solution 2: bottom up
        return self.height(root) != -1
    
    def height(self, root):
        if root == None: return 0
        left = self.height(root.left)
        right = self.height(root.right)
        if left == -1: return -1
        if right == -1: return -1
        if abs(right - left) > 1: return -1
        return max(left, right) + 1
```