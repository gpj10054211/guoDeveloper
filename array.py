import sys,re,os
import heapq,copy
import collections
from collections import deque
from typing import Optional,List
import random

class ListNode:
    def __init__(self, val = 0, next = None):
        self.val = val
        self.next = next

class solutions:
    def __init__(self, val = 0):
        self.data = [val]

    def TwoSum(self, nums: Optional[List[int]], target: int) -> Optional[List[int]]:
        map_dict = {}
        res = []
        for i in range(0, len(nums), 1):
            tmp = target - nums[i]
            if tmp in map_dict:
                 res.append(map_dict[tmp])
                 res.append(i)
                 return res
            map_dict[nums[i]] = i
        return res

    def TwoSumList(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummpy = ListNode()
        ll = dummpy.next
        bit = 0
        while l1 or l2:
            tmp = bit
            if l1:
                tmp += l1.val
                l1 = l1.next
            if l2:
                tmp += l2.val
                l2 = l2.next
            if tmp > 9:
                bit = 1
                tmp -= 10
            else:
                bit = 0
            if not ll:
                ll = ListNode(tmp)
            else:
                ll.next = ListNode(tmp)
                ll = ll.next
            if not dummpy.next:
                dummpy.next = ll

        if bit == 1:
            ll.next = ListNode(1)

        return dummpy.next

    #给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。
    def lengthOfLongestSubstring(self, s: str) -> int:
        window = collections.defaultdict(int)
        res = 0
        left = 0
        right = 0
        while right < len(s):
            d = s[right]
            #if d not in window:
                #window[d] = 0
            window[d] += 1
            right += 1

            while window[d] > 1:
                c = s[left]
                window[c] -= 1
                left += 1
            res = max(res, right - left)

        return res

    #三数之和:你返回所有和为 0 且不重复的三元组
    #输入：nums = [-1,0,1,2,-1,-4]
    #输出：[[-1,-1,2],[-1,0,1]]
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort(key=lambda x: x, reverse=False)
        res = []
        for i in range(0, len(nums), 1):
            if nums[i] > 0:
                break
            if i > 0 and nums[i] == nums[i-1]:
                continue
            left = i + 1
            right = len(nums) - 1
            while left < right:
                val = nums[left] + nums[right] + nums[i]
                if val == 0:
                    res.append([nums[i], nums[left], nums[right]])
                    while left < right and nums[left] == nums[left+1]:
                        left += 1
                    while left < right and nums[right] == nums[right-1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif val < 0:
                    left += 1
                else:
                    right -= 1
        return res

    #全排列:给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。
    #输入：nums = [1,2,3]
    #输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
    def permute(self, nums: List[int]) -> List[List[int]]:
        used = [False] * len(nums)
        res = []
        sub_res = []
        def dfs(nums: List[int]):
            if len(sub_res) == len(nums):
                res.append(sub_res[:])
            for i in range(0, len(nums), 1):
                if used[i]:
                    continue
                used[i] = True
                sub_res.append(nums[i])
                dfs(nums)
                used[i] = False
                sub_res.pop()
        dfs(nums)
        return res

    #组合总和:给你一个无重复元素的整数数组candidates 和一个目标整数target，找出candidates中可以使数字和为目标数target 的 所有不同组合 ，并以列表形式返回。你可以按 任意顺序 返回这些组合。
    #candidates 中的 同一个 数字可以无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。
    #输入：candidates = [2,3,6,7], target = 7
    #输出：[[2,2,3],[7]]
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        sub_res = []
        res = []

        def dfs(candidates: List[int], target: int, index: int, sum: int):
            if sum == target:
                res.append(sub_res[:])
            if sum > target:
                return

            for i in range(index, len(candidates), 1):
                if i > index and candidates[i] == candidates[i-1]:
                    continue
                sum += candidates[i]
                sub_res.append(candidates[i])
                dfs(candidates, target, i, sum)
                sum -= candidates[i]
                sub_res.pop()

        dfs(candidates, target, 0, 0)
        return res

    #下一个排列:
    #输入：nums = [1,2,3]
    #输出：[1,3,2]
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        if len(nums) <= 1:
            return
        j = len(nums) - 2
        while j >= 0 and nums[j] >= nums[j+1]:
            j -= 1

        if j >= 0:
            k = len(nums) - 1
            while k > j and nums[j] >= nums[k]:
                k -= 1
            nums[j],nums[k] = nums[k], nums[j]

        left = j+1
        right = len(nums) - 1
        while left <= right:
            nums[left], nums[right] = nums[right],nums[left]
            left += 1
            right -= 1

    #岛屿数量：给你一个由'1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
    #岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
    #此外，你可以假设该网格的四条边均被水包围。
    '''
    输入：grid = [
      ["1","1","1","1","0"],
      ["1","1","0","1","0"],
      ["1","1","0","0","0"],
      ["0","0","0","0","0"]
    ]
    输出：1
    '''
    def numIslands(self, grid: List[List[str]]) -> int:
         def dfs(grid: List[List[str]], i: int, j: int):
             #向上搜索
             if i > 0 and used[i-1][j] == False and grid[i-1][j] == '1':
                 used[i-1][j] = True
                 dfs(grid,i-1,j)
             #向下搜索
             if i < len(grid) - 1 and used[i+1][j] == False and grid[i+1][j] == '1':
                 used[i+1][j] = True
                 dfs(grid,i+1,j)
             #向左搜索
             if j > 0 and used[i][j-1] == False and grid[i][j-1] == '1':
                 used[i][j-1] = True
                 dfs(grid, i , j-1)
             #向右搜索
             if j < len(grid[0]) - 1 and used[i][j+1] == False and grid[i][j+1] == '1':
                 used[i][j+1] = True
                 dfs(grid, i , j+1)


         used = [[False for j in range(len(grid[0]))] for i in range(len(grid))]
         res = 0
         for i in range(len(grid)):
             for j in range(len(grid[0])):
                 if grid[i][j] == '0' or used[i][j]:
                     continue
                 used[i][j] = True
                 res += 1
                 dfs(grid, i, j)
         return res

    #打家劫舍
    #输入：[1,2,3,1]
    #输出：4
    def rob(self, nums: List[int]) -> int:
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        dp = [0] * len(nums)
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, len(nums), 1):
            dp[i] = max(dp[i-1], nums[i]+dp[i-2])
        return dp[len(nums)-1]

    #数组中的第K个最大元素
    #给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。
    #请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
    #你必须设计并实现时间复杂度为 O(n) 的算法解决此问题。
    #输入: [3,2,1,5,6,4], k = 2
    #输出: 5
    def findKthLargest(self, nums: List[int], k: int) -> int:

        #最小堆
        '''
        data = []
        for i in range(len(nums)):
            heapq.heappush(data, nums[i])
        j = 0
        while j < len(nums) - k:
            heapq.heappop(data)
            j += 1
        return heapq.heappop(data)
        '''
        data = []
        for i in range(len(nums)):
            heapq.heappush(data, nums[i])
            if len(data) > k:
                heapq.heappop(data)
        return heapq.heappop(data)


        '''
        #快排思路
        def findKth(nums: List[int], k: int) -> int:
            def partition(nums: List[int]) -> int:
                paviot = 0
                index = paviot + 1
                i = index
                while i < len(nums):
                    if nums[i] < nums[paviot]:
                        nums[i],nums[index] = nums[index],nums[i]
                        index += 1
                    i += 1
                nums[paviot],nums[index-1] = nums[index-1],nums[paviot]
                return index-1

            paviot = partition(nums)
            if paviot == k - 1:
                return nums[paviot]
            elif paviot < k - 1:
                return findKth(nums[paviot+1:], k -paviot-1)
            else:
                return findKth(nums[0:paviot], k)
        return findKth(nums,len(nums)-k+1)
        '''



    #前 K 个高频元素：给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。
    #输入: nums = [1,1,1,2,2,3], k = 2
    #输出: [1,2]
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        res = []
        map_dict = collections.defaultdict(int)
        for i in range(len(nums)):
            map_dict[nums[i]] += 1
        pairs = []

        '''
        #构造最大堆实现
        for key,value in map_dict.items():
            heapq.heappush(pairs,(-value,key))
        i = 0
        while i < k:
            tmp = heapq.heappop(pairs)
            res.append(tmp[1])
            i += 1
        '''
        #最小堆实现,维护一个包含k个值的最小堆
        for key,value in map_dict.items():
            heapq.heappush(pairs, (value,key))
            if len(pairs) > k:
                heapq.heappop(pairs)
        for i in range(k):
            res.append(heapq.heappop(pairs)[1])
        return res

    #分割等和子集
    #给你一个 只包含正整数 的 非空 数组 nums 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。
    #输入：nums = [1,5,11,5]
    #输出：true
    def canPartition(self, nums: List[int]) -> bool:
        sum = 0
        for i in range(len(nums)):
            sum += nums[i]
        if sum % 2 != 0:
            return False
        target = int(sum/2)
        dp = [[False for j in range(target + 1)] for i in range(len(nums))]
        for i in range(target+1):
            if nums[0] == i:
                dp[0][i] = True

        for i in range(1, len(nums), 1):
            for j in range(0, target+1, 1):
                if nums[i] > j:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = (dp[i-1][j] or dp[i-1][j-nums[i]])
        return dp[len(nums)-1][target]

    #计算两个日期的间隔天数
    def daydiff(self, yearst: int, monthst: int, dayst: int, yearend: int, monthend: int, dayend: int) -> int:
        def days(year: int, month: int, day: int) -> int:
             sum = 0
             months = [0,31,28,31,30,31,30,31,31,30,31,30,31]
             for i in range(month):
                 sum += months[i]
             #print (sum)
             sum += day
             if (year % 4 == 0 and year % 100 != 0 or year % 400 == 0) and month > 2:
                 sum += 1
             return sum
        print ('起始日期%d%d%d' %(yearst,monthst,dayst))
        print ('结束日期%d%d%d' %(yearend,monthend,dayend))
        totol = 0
        #先不考虑是否是闰年，按照每年有365天，计算起始日期和结束日期之间的间隔天数
        total = (yearend - yearst) * 365
        total -= days(yearst, monthst, dayst) #考虑起始日期的月和日，减去起始日期计算得到的天数
        total += days(yearend, monthend, dayend) #考虑结束日期的月和日，加上结束日期计算得到的天数

        #考虑是否是闰年，闰年需要加1
        i = yearst
        while i != yearend:
            if total > 0:
                if i % 4 == 0 and i % 100 != 0 or i % 400 == 0:
                    total += 1
                i += 1
            else:
                if i % 4 == 0 and i % 100 != 0 or i % 400 == 0:
                    i -= 1
        print ('%d年%d月%d日与%d年%d月%d日相差%d天'%(yearst,monthst,dayst, yearend,monthend,dayend,total))
        return total

    #每隔一小时发送一次消息，每次消息对应一定的ctr预估值，相邻消息发送效率降低1/5，求一天内发送消息产出ctr预估值最大的两次；
    def twoMax(self, nums: List[int]) -> List[int]:
        return []

    #实现AUC
    #y = [1,0,0,0,1,0,1,0,]
    #pred = [0.9, 0.8, 0.3, 0.1,0.4,0.9,0.66,0.7]
    #res = (sum(ranki) - M*(M+1)/2) /(M * N)
    def calc_auc(self, label: List[float], pred: List[float]) -> float:
        nums = list(zip(pred, label))
        print (nums)
        nums.sort(key = lambda x: (x[0],x[1]), reverse = False)
        print (nums)
        M = 0
        N = 0
        rank = 0
        pairs = 0
        for i in range(len(nums)):
            if nums[i][1] == 1:
                M += 1
                pairs += M
                rank += i + 1
            else:
                N += 1
        print (M, N, rank, pairs)
        return (rank - pairs)/ (M * N)

    #随机数索引：给你一个可能含有 重复元素 的整数数组 nums ，请你随机输出给定的目标数字 target 的索引。你可以假设给定的数字一定存在于数组中。
    #nums = [1, 2, 3, 3, 3]
    #target = 3
    def get_randindex(self, nums: List[int], target: int) -> int:
        count = 0
        res = -1
        for i in range(len(nums)):
            if nums[i] == target:
                if random.randint(0,count) == 0:
                    res = i
                count += 1
        return res

    #区间列表的交集：给定两个由一些闭区间组成的列表，firstList 和 secondList ，其中 firstList[i] = [starti, endi] 而 secondList[j] = [startj, endj]。
    #每个区间列表都是成对不相交的，并且已经排序。返回这两个区间列表的交集。
    #firstList = [[0,2],[5,10],[13,23],[24,25]]
    #secondList = [[1,5],[8,12],[15,24],[25,26]]
    firstList = [[14,16]]
    secondList = [[7,13],[16,20]]
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        res = []
        '''
        暴力解法
        for i in range(len(firstList)):
            for j in range(len(secondList)):
                if secondList[j][0] > firstList[i][1]:
                    break
                if secondList[j][1] < firstList[i][0]:
                    continue
                if firstList[i][0] <= secondList[j][0] and firstList[i][1] >= secondList[j][1]:
                    res.append([secondList[j][0],secondList[j][1]])
                elif firstList[i][0] >= secondList[j][0] and firstList[i][1] <= secondList[j][1]:
                    res.append([firstList[i][0],firstList[i][1]])
                else:
                    res.append([max(firstList[i][0],secondList[j][0]), min(firstList[i][1],secondList[j][1])])
        '''
        for i in range(len(firstList)):
            for j in range(len(secondList)):
                start = max(firstList[i][0], secondList[j][0])
                end = min(firstList[i][1], secondList[j][1])
                if start <= end:
                    res.append([start,end])

        #双指针法：
        f = 0
        s = 0
        while f < len(firstList) and s < len(secondList):
            start = max(firstList[f][0], secondList[s][0])
            end = min(firstList[f][1], secondList[s][1])
            if start <= end:
                res.append([start,end])
            if firstList[f][1] < secondList[s][1]:
                f += 1
            else:
                s += 1

        return res

    #合并区间：以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。
    #输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
    #输出：[[1,6],[8,10],[15,18]]
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if len(intervals) <= 1:
            return intervals
        intervals.sort(key=lambda x: (x[0], x[1]), reverse = False)
        start = intervals[0][0]
        end = intervals[0][1]
        res = []
        for i in range(1, len(intervals), 1):
            if intervals[i][0] > end:
                res.append([start,end])
                start = intervals[i][0]
                end = intervals[i][1]
            else:
                if intervals[i][1] >= end:
                     end = intervals[i][1]
        res.append([start, end])
        return res


    #最小路径和：输入类二叉树的列表，列表中的每个元素都是下一行对应两个元素的父节点，求从起始节点到叶子节点的最小路径和
    #输入： [[1],
   #       [3,2],
   #      [4,5,6],
   #     [7,8,6,9]]
    #输出：1 + 2 + 5 + 6 = 14
    def min_pathsum(self, input: List[List[int]]) -> int:
        dp = copy.deepcopy(input)
        res = sys.maxsize
        for i in range(len(input)):
            for j in range(len(input[i])):
                if i == 0:
                    dp[i][j] = input[i][j]
                else:
                    if j == 0:
                        dp[i][j] += dp[i-1][j]
                    elif j < len(input[i]) - 1:
                        dp[i][j] += min(dp[i-1][j],dp[i-1][j-1])
                    else:
                        dp[i][j] += dp[i-1][j-1]
                #print (i,j,dp[i][j], input[i][j])


                if i == len(input) - 1:
                    res = min(res, dp[i][j])
        #print (dp)

        return res

    #字符串多余空格处理，首尾不含空格，中间连续空格只保留一个
    #输入："  sa   bb   "
    #输出："sa bb"
    def del_space(self, input: List[str]) -> List[str]:
        left = 0
        right = 0
        while right < len(input):
            if input[right] != ' ':
                input[left] = input[right]
                left += 1
                right += 1
            else:
                if right > 0:
                    if input[right-1] != ' ':
                        input[left] = input[right]
                        left += 1
                right += 1


        if input[0: left][-1] == ' ':
            return input[0: left-1]

        return input[0: left]





if __name__ == '__main__':
    ss = solutions()
    #input = [[1],[3,2],[4,5,6],[7,8,6,9]]
    #print (ss.min_pathsum(input))

    input = "  sa   bb   "
    input  = list(input)
    print (input)
    print (ss.del_space(input))



    #firstList = [[14,16]]
    #secondList = [[7,13],[16,20]]
    #print (ss.intervalIntersection(firstList,secondList))



    #y = [1,0,0,0,1,0,1,0,]
    #pred = [0.9, 0.8, 0.3, 0.1,0.4,0.9,0.66,0.7]
    #print (ss.calc_auc(y, pred))

    #nums = [1, 2, 3, 3, 3]
    #target = 3
    #i = 0
    #map_dict = collections.defaultdict(int)
    #while i < 10000:
        #map_dict[ss.get_randindex(nums, target)] += 1
        #i += 1
    #for key, value in map_dict.items():
        #print (key,value)



    #nums = [1,4,2,9,7,10,6]
    #target = 13
    #print (ss.TwoSum(nums, target))

    #s = "abcdsc"
    #print (ss.lengthOfLongestSubstring(s))

    #nums = [-1,0,1,2,-1,-4]
    #print (ss.threeSum(nums))

    #nums = [1,2,3]
    #print (ss.permute(nums))

    #candidates = [2,3,6,7]
    #target = 7
    #print (ss.combinationSum(candidates,target))

    #nums = [3,1,2]
    #ss.nextPermutation(nums)
    #print (nums)

    '''
    grid = [
          ["1","1","1","1","0"],
          ["1","1","0","1","0"],
          ["1","1","0","0","0"],
          ["0","0","0","0","0"]
        ]
    print (ss.numIslands(grid))
    '''
    #nums = [1,2,3,1]
    #print (ss.rob(nums))

    #nums = [3,2,1,5,6,4]
    #k = 2
    #print (ss.findKthLargest(nums,k))

    #nums = [1,1,1,2,2,3]
    #k = 2
    #print (ss.topKFrequent(nums,k))

    #nums = [1,5,11,5]
    #print (ss.canPartition(nums))

    #yearst = 2019
    #monthst = int('8')
    #dayst = int(13)

    #yearend = 2020
    #monthend = int('9')
    #dayend = int(20)

    #print (ss.daydiff(yearst,monthst,dayst,yearend,monthend,dayend))







