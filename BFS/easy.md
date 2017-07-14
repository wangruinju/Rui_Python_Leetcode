---
title: "Easy"
author: "Rui Wang"
date: "7/8/2017"
output: html_document
---

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

107. Binary Tree Level Order Traversal II

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        res = []
        if not root: return res
        queue = [root]
        while queue:
            # how many node in the same level
            n = len(queue)
            sublevel = []
            # put all nodes from the next level in the queue
            for i in range(n):
                if queue[-1].left:
                    queue.insert(0, queue[-1].left)
                if queue[-1].right:
                    queue.insert(0, queue[-1].right)
                sublevel.append(queue.pop().val)
            # not like 102, we put the each level in the front just like a queue
            res.insert(0, sublevel)
        return res
```
