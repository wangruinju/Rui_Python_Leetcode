---
title: "Easy"
author: "Rui Wang"
date: "7/8/2017"
output: html_document
---

401. Binary Watch

```python 
class Solution(object):
    def readBinaryWatch(self, num):
        """
        :type num: int
        :rtype: List[str]
        """
        res = []
        for h in xrange(12):
            for m in xrange(60):
                if (bin(h) + bin(m)).count('1') == num:
                    res.append("%d:%02d"%(h,m))
        return res
```