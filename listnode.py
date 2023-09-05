import heapq,copy,collections
from typing import List,Optional
from collections import deque

class ListNode:
    def __init__(self, val = 0, next = None):
        self.val = val
        self.next = next

class Solution:

    #排序链表:给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        def mergeTwoList(head1: Optional[ListNode], head2: Optional[ListNode]) -> Optional[ListNode]:
            dummpy = ListNode()
            head = dummpy
            while head1 and head2:
                if head1.val < head2.val:
                    head.next = ListNode(head1.val)
                    head1 = head1.next
                else:
                    head.next = ListNode(head2.val)
                    head2 = head2.next
                head = head.next
            if head1:
                head.next = head1
            if head2:
                head.next = head2
            return dummpy.next

        if not head or not head.next:
            return head
        slow = head
        fast = head.next.next
        while slow and fast:
            if not fast.next:
                break
            fast = fast.next.next
            slow = slow.next

        mid_node = slow.next
        slow.next = None
        return mergeTwoList(self.sortList(head), self.sortList(mid_node))

    #相交链表:给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 null
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
            ka = 0
            kb = 0
            tmpA = headA
            tmpB = headB
            while tmpA:
                ka += 1
                tmpA = tmpA.next
            while tmpB:
                kb += 1
                tmpB = tmpB.next

            tmpA = headA
            tmpB = headB
            while ka > kb and tmpA:
                tmpA = tmpA.next
                ka -= 1
            while ka < kb and tmpB:
                tmpB = tmpB.next
                kb -= 1

            while tmpA and tmpB:
                if tmpA == tmpB:
                    return tmpA
                tmpA = tmpA.next
                tmpB = tmpB.next
            return None

    #环形链表II：给定一个链表的头节点  head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
            if not head or not head.next:
                return None
            slow = head
            fast = head
            while slow and fast:
                if not fast.next:
                    return None
                slow = slow.next
                fast = fast.next.next
                if slow == fast:
                    break


            slow = head
            while slow and fast:
                if slow == fast:
                    return slow
                slow = slow.next
                fast = fast.next
            return None

    #反转链表：给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        tail = head
        while tail.next:
            cur_node = tail.next
            tail.next = cur_node.next
            cur_node.next = head
            head = cur_node
        return head



