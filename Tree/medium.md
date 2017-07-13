---
title: "Medium"
author: "Rui Wang"
date: "7/8/2017"
output: html_document
---

222. Count Complete Tree Nodes

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def countNodes(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # O(N) is out of time
        # if not root: return 0
        # return 1 + self.countNodes(root.left) + self.countNodes(root.right)
        
        if not root: return 0
        left = right = root
        height = 0
        while right:
            left = left.left
            right = right.right
            height += 1
        if not left:
            # 1 2 4 ... 2^(height-1) the sum is 2^height-1
            return (1 << height) - 1
        return 1 + self.countNodes(root.left) + self.countNodes(root.right)
```

199. Binary Tree Right Side View

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        # solution 1 use queue
        # res = []
        # if not root: return res
        # queue = [root]
        # while queue:
        #     n = len(queue)
        #     res.append(queue[0].val)
        #     for i in range(n):
        #         node = queue.pop()
        #         if node.left:
        #             queue.insert(0, node.left)
        #         if node.right:
        #             queue.insert(0, node.right)
        # return res
        
        # solution 2 use dfs
        
        self.res = []
        def helper(root, level):
            if not root: return []
            if level == len(self.res):
                self.res.append(root.val)
            helper(root.right, level + 1)
            helper(root.left, level + 1)
        helper(root, 0)
        return self.res
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

129. Sum Root to Leaf Numbers

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sumNumbers(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # solution 1
        # dfs + stack
        if not root:
            return 0
        stack, res = [(root, root.val)], 0
        while stack:
            node, value = stack.pop()
            if node:
                if not node.left and not node.right:
                    res += value
                if node.right:
                    stack.append((node.right, value*10+node.right.val))
                if node.left:
                    stack.append((node.left, value*10+node.left.val))
        return res
        
        # solution 2
        # bfs + queue
        # if not root:
        #     return 0
        # queue, res = collections.deque([(root, root.val)]), 0
        # while queue:
        #     node, value = queue.pop(0)
        #     if node:
        #         if not node.left and not node.right:
        #             res += value
        #         if node.left:
        #             queue.append((node.left, value*10+node.left.val))
        #         if node.right:
        #             queue.append((node.right, value*10+node.right.val))
        # return res
    
        # solution 3
        # recursively 
#         self.res = 0

#         def dfs(root, value):
#             if root:
#                 if not root.left and not root.right:
#                     self.res += value*10 + root.val
#                 dfs(root.left, value*10+root.val)
#                 dfs(root.right, value*10+root.val)

#         dfs(root, 0)
#         return self.res
```

117. Populating Next Right Pointers in Each Node II

```python
# Definition for binary tree with next pointer.
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None

class Solution:
    # @param root, a tree link node
    # @return nothing
    def connect(self, root):
        while root:
            dummy = TreeLinkNode(0)
            cur = dummy
            while root:
                if root.left:
                    cur.next = root.left
                    cur = cur.next
                if root.right:
                    cur.next = root.right
                    cur = cur.next
                root = root.next
            # reach the next level
            root = dummy.next
```

116. Populating Next Right Pointers in Each Node

```python
# Definition for binary tree with next pointer.
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None

class Solution:
    # @param root, a tree link node
    # @return nothing
    def connect(self, root):
        if not root:
            return
        pre = root
        cur = None
        while pre.left:
            cur = pre
            while cur:
                cur.left.next = cur.right
                if cur.next:
                    cur.right.next = cur.next.left
                cur = cur.next
            pre = pre.left
```

114. Flatten Binary Tree to Linked List

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        """
        # solution 1
        # top - down
        # if not root: return
        # cur = root
        # while cur:
        #     if cur.left:
        #         pre = cur.left
        #         while pre.right:
        #             pre = pre.right
        #         pre.right = cur.right
        #         cur.right = cur.left
        #         cur.left = None
                
        #     cur = cur.right
        
        # solution 2
        # dfs bottom - up
        self.pre = None
        def dfs(root):
            if not root: return
            dfs(root.right)
            dfs(root.left)
            
            root.right = self.pre
            root.left = None
            self.pre = root
        dfs(root)
```

113. Path Sum II

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
        :rtype: List[List[int]]
        """
        self.res = []

        if not root:
            return self.res
        
        
        def dfs(root, sum, stack):
            stack.append(root.val)
            if not root.left and not root.right and sum == root.val:
                self.res.append(stack[:])
            if root.left:
                dfs(root.left, sum-root.val, stack)
            if root.right:
                dfs(root.right, sum-root.val, stack)
            stack.pop()
        
        dfs(root, sum, [])
        
        return self.res
```

106. Construct Binary Tree from Inorder and Postorder Traversal

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        # postorder: left right root
        # inorder left root right
        if not inorder or not postorder:
            return None
        root = TreeNode(postorder.pop())
        ind = inorder.index(root.val)
        root.right = self.buildTree(inorder[ind+1:], postorder)
        root.left = self.buildTree(inorder[:ind], postorder)
        return root
```

105. Construct Binary Tree from Preorder and Inorder Traversal

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        # preorder: root left right
        # inorder: left root right
        if not inorder or not preorder:
            return None
        
        root = TreeNode(preorder.pop(0))
        ind = inorder.index(root.val)
        # forward build tree
        # left right
        root.left = self.buildTree(preorder, inorder[:ind])
        root.right = self.buildTree(preorder, inorder[ind+1:])

        return root
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

102. Binary Tree Level Order Traversal

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        # solution 1 DFS
        # attach the value at each level
        # if the height is larger than the list size, add [] for another level
    #     res = []
    #     self.levelhelper(res, root, 0)
    #     return res
        
    # def levelhelper(self, res, root, height):
    #     if root:
    #         if height >= len(res):
    #             res.append([])
    #         res[height].append(root.val)
    #         self.levelhelper(res, root.left, height + 1)
    #         self.levelhelper(res, root.right, height + 1)
    
        # solution 2 using queue
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
            res.append(sublevel)
        return res 
```

515. Find Largest Value in Each Tree Row

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def largestValues(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        # it is the same problem with 102, 107
        # just add the max into res when treating the sublist
        # however, it take too many spaces
        # do not apply sublist, apply a temp max, make comparison everytime and attach the max value for each level

        res = []
        if not root: return []
        queue = [root]
        
        while queue:
            n = len(queue)
            # python only have sys.maxint for maximum of integer
            # we will use float('-inf') and float('inf')
            submax = float('-inf')
            for i in range(n):
                node  = queue.pop()
                submax = max(submax, node.val)
                if node.left:
                    queue.insert(0, node.left)
                if node.right:
                    queue.insert(0, node.right)
            res.append(submax)
        return res
        
        # solution 2
        # use the level, dfs
    #     self.res = []
    #     self.helper(root, 0)
    #     return self.res
    
    # def helper(self, root, level):
    #     if root:
    #         if level >= len(self.res):
    #             self.res.append(root.val)
    #         self.res[level] = max(self.res[level], root.val)
    #         self.helper(root.left, level + 1)
    #         self.helper(root.right, level + 1)        
```

98. Validate Binary Search Tree

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True
        stack = []
        pre = None
        
        while root or stack:
            # go the smallest node
            while root:
                stack.append(root)
                root = root.left
            
            root = stack.pop()
            if pre and pre.val >= root.val:
                return False
            pre = root
            root = root.right
        return True
```

96. Unique Binary Search Trees

```python
class Solution(object):
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        # for different root
        # if the root is i
        # then i, i+2 ... n will be on the right subtree, the count is hash[n-i]
        # the left subtree consists of 1, 2, i-1, the count is hash[i-1]
        # multiple for just root i
        hash = {0:1}
        index = 0
        while index <= n:
            sum = 0
            if index not in hash.keys():
                for i in range(index):
                    sum += hash[i]*hash[index-1-i]
                hash[index] = sum
            index += 1
        return hash[n]  
```

95. Unique Binary Search Trees II

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def generateTrees(self, n):
        """
        :type n: int
        :rtype: List[TreeNode]
        """
        if n == 0: return []
        return self.dfs(1, n)
    
    def dfs(self, start, end):
        if start > end: return [None]
        res = []
        for rootval in range(start, end+1):
            lefttree = self.dfs(start, rootval-1)
            righttree = self.dfs(rootval+1, end)
            for i in lefttree:
                for j in righttree:
                    # preorder: root left right
                    root = TreeNode(rootval)
                    root.left = i
                    root.right = j
                    res.append(root)
        return res  
```

230. Kth Smallest Element in a BST

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        # solution 1 
        # use the property of bst
        # take all elements in a array orderly
        # return the kth element
        # it take a lot of spaces if the tree has too many elements
    #     res = []
    #     self.helper(root, res)
    #     return res[k-1]
    
    # def helper(self, node, res):
    #     if not node:
    #         return 
    #     self.helper(node.left, res)
    #     res.append(node.val)
    #     self.helper(node.right, res)
    
        # solution 2
        # binary search 
        # repeatedly count the tree on the left
    #     count = self.count(root.left)
    #     # if left count is larger or equal than k, pass to the left node
    #     if count >= k:
    #         return self.kthSmallest(root.left, k)
    #     # count+1 left+current node
    #     elif count+1 < k:
    #         return self.kthSmallest(root.right, k-count-1)
    #     # if count+1 == k
    #     return root.val
        
    # def count(self, root):
    #     if not root:
    #         return 0
    #     return 1+self.count(root.left)+self.count(root.right)
        
        # solution 3
        # dfs in order: left root right
        # self.res = 0
        # self.count = k
        # def dfs(root):
        #     if not root:
        #         return 0
        #     dfs(root.left)
        #     self.count -=1
        #     if self.count == 0:
        #         self.res = root.val
        #         return
        #     dfs(root.right)
    
        # dfs(root)
        # return self.res
        
        # solution 4
        # use stack
        stack = []
        
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            k -= 1
            if k == 0: 
                break
            root = root.right
        return root.val
```

513. Find Bottom Left Tree Value

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def findBottomLeftValue(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        queue = [root]
        while queue:
            node = queue.pop()
            
            if node.right:
                queue.insert(0, node.right)
            if node.left:
                queue.insert(0, node.left)
                
        return node.val
```

508. Most Frequent Subtree Sum

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def findFrequentTreeSum(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        # use self.hash and self.mode
        if root == None: return []
        self.hash = {}
        self.mode = 0
        res = []
        def getsum(root):
            # left right root
            if not root: return 0
            left = getsum(root.left)
            right = getsum(root.right)
            sum = root.val+left+right
            
            if sum in self.hash:
                self.hash[sum] += 1
            else:
                self.hash[sum] = 1
            
            self.mode = max(self.mode, self.hash[sum])
            return sum
            
        getsum(root)

        for s in self.hash.keys():
            if self.hash[s] == self.mode:
                res.append(s)
        return res
```

450. Delete Node in a BST

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def deleteNode(self, root, key):
        """
        :type root: TreeNode
        :type key: int
        :rtype: TreeNode
        """
        if not root:
            return None
        
        if key < root.val:
            root.left = self.deleteNode(root.left, key)
        elif key > root.val:
            root.right = self.deleteNode(root.right, key)
        elif not root.left:
            return root.right
        elif not root.right:
            return root.left
        else:
            rightsmall = root.right
            while rightsmall.left:
                rightsmall = rightsmall.left
            rightsmall.left = root.left
            return root.right
#         else:
#             root.val = self.helper(root.right)
#             root.right = self.deleteNode(root.right, root.val)
        
        return root
    
#     def helper(self, node):
#         while node.left:
#             node = node.left
#         return node.val
```

449. Serialize and Deserialize BST

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        # root left right
        if not root: return ''
        left = self.serialize(root.left)
        right = self.serialize(root.right)
        res = str(root.val)
        if left: res += ',' + left
        if right: res += ',' + right
        return res

        
    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        if not data: return None
        
        nums = map(int, data.split(','))
        queue = []
        for n in nums:
            queue.append(n)
        
        return self.helper(queue)
    
    def helper(self, queue):
        if not queue:
            return None
        
        root = TreeNode(queue.pop(0))
        leftqueue = []
        while queue and queue[0] < root.val:
            leftqueue.append(queue.pop(0))
        root.left = self.helper(leftqueue)
        root.right = self.helper(queue)
        return root
        

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))
```

337. House Robber III

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def rob(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # solution 1 overtime not pass 
        # if not root: return 0
        # sum = 0
        # if root.left:
        #     sum += self.rob(root.left.left) + self.rob(root.left.right)
        # if root.right:
        #     sum += self.rob(root.right.left) + self.rob(root.right.right)
        # return max(sum+root.val, self.rob(root.left) + self.rob(root.right))
        
        # change it to dict and store it
        
        return self.robsub(root, {})
    def robsub(self, root, map):
        if not root:
            return 0
        if root in map:
            return map[root]
        sum = 0
        if root.left:
            sum += self.robsub(root.left.left, map) + self.robsub(root.left.right, map)
        if root.right:
            sum += self.robsub(root.right.left, map) + self.robsub(root.right.right, map)
        sum = max(sum+root.val, self.robsub(root.left, map) + self.robsub(root.right, map))
        map[root] = sum
        
        return sum
    
        # solution 2 greedy dfs
#         def helper(root):
#             if not root:
#                 return [0,0]
            
#             res = [0,0]
#             left = helper(root.left)
#             right = helper(root.right)
#             res[0] = max(left[0], left[1]) + max(right[0], right[1])
#             res[1] = root.val + left[0] + right[0]
#             return res
            
#         res = helper(root)
#         return max(res[0], res[1])
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

236. Lowest Common Ancestor of a Binary Tree

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
        if root in (None, p, q): return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        else:
            return left or right
```

623. Add One Row to Tree

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def addOneRow(self, root, v, d):
        """
        :type root: TreeNode
        :type v: int
        :type d: int
        :rtype: TreeNode
        """        
        # solution 1
        # very nice code
        # dummy = TreeNode(None)
        # dummy.left = root
        # row = [dummy]
        # for _ in range(d - 1):
        #     # traverse
        #     row = [kid for node in row for kid in (node.left, node.right) if kid]
        # for node in row:
        #     node.left, node.left.left = TreeNode(v), node.left
        #     node.right, node.right.right = TreeNode(v), node.right
        # return dummy.left
        
        # solution 2 bfs using queue
#         if d == 1:
#             node = TreeNode(v)
#             node.left = root
#             return node

#         queue = [root]
#         for i in range(d-2):
#             for j in range(len(queue)):
#                 temp = queue.pop(0)
#                 if temp.left: queue.append(temp.left)
#                 if temp.right: queue.append(temp.right)

#         while queue:
#             temp = queue.pop(0)
#             temp.left, temp.left.left = TreeNode(v), temp.left
#             temp.right, temp.right.right = TreeNode(v), temp.right

#         return root
    
        # solution 3 dfs
#         if d == 1:
#             node = TreeNode(v)
#             node.left = root
#             return node
        
#         def dfs(root, depth, v, d):
#             if not root: return None
#             if depth < d-1:
#                 dfs(root.left, depth+1, v, d)
#                 dfs(root.right, depth+1, v, d)
#             else:
#                 root.left, root.left.left = TreeNode(v), root.left
#                 root.right, root.right.right = TreeNode(v), root.right
#         dfs(root, 1, v, d)
        
#         return root
    
        # solution 4 dfs put v to the left when d=1 else the right if d=0
        if d < 2:
            node = TreeNode(v)
            if d:
                node.left = root
            else:
                node.right = root
            return node
        
        if not root: return None
        root.left = self.addOneRow(root.left, v, 1 if d ==2 else d-1)
        root.right = self.addOneRow(root.right, v, 0 if d ==2 else d-1)
        return root
```