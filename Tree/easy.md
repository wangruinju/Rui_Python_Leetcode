---
title: "Easy"
author: "Rui Wang"
date: "7/8/2017"
output: html_document
---

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

235. Lowest Common Ancestor of a Binary Search Tree

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if not root or not p or not q:
            return None
        if max(p.val, q.val) < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        if min(p.val, q.val) > root.val:
            return self.lowestCommonAncestor(root.right, p, q)
        return root
```

606. Construct String from Binary Tree

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def tree2str(self, t):
        """
        :type t: TreeNode
        :rtype: str
        """
        
        if not t:
            return ''
        elif not t.left and not t.right:
            return str(t.val)
        elif not t.left and t.right:
            return str(t.val) + "(" + ")" + "(" + self.tree2str(t.right) + ")"
        elif t.left and not t.right:
            return str(t.val) + "(" + self.tree2str(t.left) + ")"
        else:
            return str(t.val) + "(" + self.tree2str(t.left) + ")" + "(" + self.tree2str(t.right) + ")"
```

501. Find Mode in Binary Search Tree

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def findMode(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        # apply dict to cal each cnt when tranversing
        dict = {}
        res = []
        
        if not root:
            return res
        
        def buildhash(root, dict):
            if root:
                if root.val in dict:
                    dict[root.val] += 1
                else:
                    dict[root.val] = 1
                buildhash(root.left, dict)
                buildhash(root.right, dict)
                
        
        buildhash(root, dict)
        
        max_modes = max(dict.values())
        
        for key in dict.keys():
            if dict[key] == max_modes:
                res.append(key)
        return res         
```

543. Diameter of Binary Tree

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    # the longest length is not from outside
    # set a global best can make the algorithm down to O(n)

    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.best = 0
        
        self.depth(root)
        return self.best
        
    def depth(self, root):
    
        if not root:
            return 0
            
        left = self.depth(root.left)
        right = self.depth(root.right)
        self.best = max(self.best, left + right)
        
        return max(left, right) + 1         
```

437. Path Sum III

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: int
        """
        # solution 1 recursive take O(N^2)
    #     if root:
    #         return self.findpath(root, sum) + self.pathSum(root.left, sum) + self.pathSum(root.right, sum)
    #     return 0
        
    # def findpath(self, root, sum):
    #     if root:
    #         return int(root.val == sum) + self.findpath(root.left, sum-root.val) + self.findpath(root.right, sum-root.val)
    #     return 0
        # solution 2 use hash table to count
        
        return self.helper(root, 0, sum, {0:1})
        
    def helper(self, root, cursum, target, dict):
        # if root none, the count 0
        if not root: return 0
        cursum += root.val
        # if not exist, set 0
        self.res = dict.setdefault(cursum - target, 0)
        # put cursum into the dict, if exit, add one to the count
        dict[cursum] = dict.setdefault(cursum, 0) + 1
        # cursum updated and target for root.left and root.right 
        self.res += self.helper(root.left, cursum, target, dict) + self.helper(root.right, cursum, target, dict)
        # since we have traversed root node, set it back means that we will not use the cursum from root node
        # if there exists cursum from down side, we can still use it
        dict[cursum] -= 1
        return self.res 
```

404. Sum of Left Leaves

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sumOfLeftLeaves(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        res = 0
        if root.left:
            if not root.left.left and not root.left.right:
                res += root.left.val
            else: 
                res += self.sumOfLeftLeaves(root.left)
        res += self.sumOfLeftLeaves(root.right)
        return res           
```

538. Convert BST to Greater Tree

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def convertBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        # if not root:
        #     return root
        # sum = 0
        # node = root
        # stack = []
        # while stack or root:
        #     while root:
        #         stack.append(root)
        #         root = root.right
        #     root = stack.pop()
        #     sum += root.val
        #     root.val = sum
        #     root = root.left
        # return node
        
        # order right root left
        self.sum = 0

        def dfs(root):
            if not root: return
            dfs(root.right)
            root.val += self.sum
            self.sum = root.val
            dfs(root.left)
        dfs(root)
        return root
```

637. Average of Levels in Binary Tree

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """
        # solution 1
        
        # use bfs
#         res = []
#         if not root:
#             return res
        
#         queue = []
#         queue.append(root)
#         while queue:
#             n = len(queue)
#             sum = 0.0
#             for i in range(n):
#                 node = queue.pop(0)
#                 sum += node.val
#                 if node.left:
#                     queue.append(node.left)
#                 if node.right:
#                     queue.append(node.right)
#             res.append(sum/n)
#         return res
            
        # solution 2
        # use dfs
        count = []
        def dfs(node, depth = 0):
            if node:
                if len(count) <= depth:
                    count.append([0, 0])
                # a list: sum and count (1 or 2)
                count[depth][0] += node.val
                count[depth][1] += 1
                dfs(node.left, depth + 1)
                dfs(node.right, depth + 1)
        dfs(root)

        return [s/float(c) for s, c in count]   
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

226. Invert Binary Tree

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        # solution 1
        # if root:
        # # swap the left and right nodes
        
        #     temp = root.left
        #     root.left = root.right
        #     root.right = temp
        #     # recursion
        #     self.invertTree(root.left)
        #     self.invertTree(root.right)
        #     return root
        
        # solution 2
        # use stack
        stack = [root]
        while stack:
            node = stack.pop()
            if node:
                # switch
                node.left, node.right = node.right, node.left
                # go the next node
                stack += node.left, node.right
        return root
```

572. Subtree of Another Tree

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSubtree(self, s, t):
        """
        :type s: TreeNode
        :type t: TreeNode
        :rtype: bool
        """
    #     # solution 1
    #     # s and t all nonempty bt
    #     if not s or not t:
    #         return False
        
    #     if self.issametree(s, t):
    #         return True
        
    #     return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)
    
    # def issametree(self, p, q):
    #     if not p and not q:
    #         return True
    #     if not p or not q:
    #         return False
    #     if p.val != q.val:
    #         return False
    #     return self.issametree(p.left, q.left) and self.issametree(p.right, q.right)
    
    # solution 2: traverse and append string, check whether string s contains string t
        def tostring(root):
            return "^" + str(root.val) + tostring(root.left) + tostring(root.right) if root else "#"
            
        return tostring(t) in tostring(s)  
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

563. Binary Tree Tilt

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def findTilt(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # first solution
        # recursive too slow O(N^2)
    #     if not root:
    #         return 0
    #     self.res = 0
    #     self.res += self.helper(root) + self.findTilt(root.left) + self.findTilt(root.right)
    #     return self.res
        
    # def helper(self, root):
    #     if not root:
    #         return 0
    #     return abs(self.rootsum(root.left) - self.rootsum(root.right))
        
    # def rootsum(self, root):
    #     if not root:
    #         return 0
    #     res = root.val
    #     res += self.rootsum(root.left) + self.rootsum(root.right)
    #     return res
        # solution 2 using postorder
        self.res = 0
        self.postorder(root)
        return self.res
        # function postorder calculate the sum like rootsum in the first solution
        # the good thing is add each root tilt when traverse
    def postorder(self, root):
        if not root: return 0
        left = self.postorder(root.left)
        right = self.postorder(root.right)
        self.res += abs(left - right)
        return left + right + root.val    
```

617. Merge Two Binary Trees

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def mergeTrees(self, t1, t2):
        """
        :type t1: TreeNode
        :type t2: TreeNode
        :rtype: TreeNode
        """
        # if not t1 and not t2: return None
        # ans = TreeNode((t1.val if t1 else 0) + (t2.val if t2 else 0))
        # ans.left = self.mergeTrees(t1 and t1.left, t2 and t2.left)
        # ans.right = self.mergeTrees(t1 and t1.right, t2 and t2.right)
        # return ans
        
        if t1 and t2:
            root = TreeNode(t1.val + t2.val)
            root.left = self.mergeTrees(t1.left, t2.left)
            root.right = self.mergeTrees(t1.right, t2.right)
            return root
        else:
            return t1 or t2
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