import sys,re,os,heapq,copy
import collections
from collections import deque

class LinkedlistNode:
    def __init__(self, key=0, value=0,pre=None, next=None):
        self.key = key
        self.value=value
        self.pre = pre
        self.next = next

class LRUCache:
    def __init__(self, capacity: int):
        self.size = 0
        self.capacity = capacity
        self.lru = dict()
        self.head = LinkedlistNode()
        self.tail = LinkedlistNode()
        self.head.next = self.tail
        self.tail.pre = self.head

    def get(self, key: int) -> int:
        if key not in self.lru:
            return -1
        node = self.lru[key]
        #删除node节点
        node.pre.next = node.next
        node.next.pre = node.pre
        #移到头结点
        node.pre = self.head
        node.next = self.head.next
        self.head.next.pre = node
        self.head.next = node
        return node.value

    def put(self, key: int, value: int) -> None:
        if key in self.lru:
            node = self.lru[key]
            node.value = value
            #删除node节点
            node.pre.next = node.next
            node.next.pre = node.pre
            #移到头结点
            node.pre = self.head
            node.next = self.head.next
            self.head.next.pre = node
            self.head.next = node
        else:
            node = LinkedlistNode(key, value)
            self.lru[key] = node
            self.size += 1
            #移到头结点
            node.pre = self.head
            node.next = self.head.next
            self.head.next.pre = node
            self.head.next = node
            if self.size > self.capacity:
                #移除尾节点
                self.size -= 1
                tmp_node = self.tail.pre
                tmp_node.pre.next = tmp_node.next
                tmp_node.next.pre = tmp_node.pre
                self.lru.pop(tmp_node.key)



