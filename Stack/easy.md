---
title: "Easy"
author: "Rui Wang"
date: "7/8/2017"
output: html_document
---

225. Implement Stack using Queues

```python
class MyStack(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.queue = []
        

    def push(self, x):
        """
        Push element x onto stack.
        :type x: int
        :rtype: void
        """
        self.queue.append(x)
        

    def pop(self):
        """
        Removes the element on top of the stack and returns that element.
        :rtype: int
        """
        if self.queue:
            # only need len(self.queue)-1 steps
            for i in range(len(self.queue)-1):
                self.queue.append(self.queue.pop())
            return self.queue.pop()
        else:
            return []
        

    def top(self):
        """
        Get the top element.
        :rtype: int
        """
        # same idea with pop, need one more step
        if self.queue:
            temp = 0
            for i in range(len(self.queue)):
                temp = self.queue.pop()
                self.queue.append(temp)
            return temp
        else:
            return []
        
        

    def empty(self):
        """
        Returns whether the stack is empty.
        :rtype: bool
        """
        return False if self.queue else True
        



# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()
```

20. Valid Parentheses

```python
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        for string in s:
            if string in "([{":
                stack.append(string)
            if string == ")":
                if not stack or stack.pop() != "(":
                    return False
            if string == "]":
                if not stack or stack.pop() != "[":
                    return False
            if string == "}":
                if not stack or stack.pop() != "{":
                    return False
        
        return False if stack else True
```

155. Min Stack

```python
class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        # set up two stacks
        self.stack = []
        self.minstack = []

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        self.stack.append(x)
        # always keep the min on the top of minstack
        if self.minstack:
            x = min(x, self.minstack[-1])
        self.minstack.append(x)
        
    def pop(self):
        """
        :rtype: void
        """
        self.minstack.pop()
        self.stack.pop()

    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1]
        

    def getMin(self):
        """
        :rtype: int
        """
        return self.minstack[-1]


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
```

232. Implement Queue using Stacks

```python
class MyQueue(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.stack = []
        

    def push(self, x):
        """
        Push element x to the back of queue.
        :type x: int
        :rtype: void
        """
        self.stack.insert(0, x)
        

    def pop(self):
        """
        Removes the element from in front of queue and returns that element.
        :rtype: int
        """
        if self.stack:
            temp = self.stack[-1]
            self.stack = self.stack[:len(self.stack)-1]
            return temp
        else:
            return []
            
        

    def peek(self):
        """
        Get the front element.
        :rtype: int
        """
        if self.stack:
            return self.stack[-1]
        else:
            return []
        
        

    def empty(self):
        """
        Returns whether the queue is empty.
        :rtype: bool
        """
        return self.stack == []
        


# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()
```

496. Next Greater Element I

```python
class Solution(object):
    def nextGreaterElement(self, findNums, nums):
        """
        :type findNums: List[int]
        :type nums: List[int]
        :rtype: List[int]
        """
        # solution 1 using two loops
        # dict = {}
        # for i in xrange(len(nums)-1):
        #     for j in xrange(i+1, len(nums)):
        #         if nums[j] > nums[i]:
        #             dict[nums[i]] = nums[j]
        #             break
        # res = []
        # for n in findNums:
        #     if n in dict:
        #         res.append(dict[n])
        #     else:
        #         res.append(-1)
        # return res
        
        # solution 2 using stack
        dict = {}
        stack = []
        for num in nums:
            while stack and stack[-1]<num:
                dict[stack.pop()] = num
            stack.append(num)
        res = []
        for num in findNums:
            if num in dict:
                res.append(dict[num])
            else:
                res.append(-1)
        return res
```
