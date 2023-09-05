
#反转字符串中的单词：
def reverse_str(input: str):
    input = list(input)
    left = 0
    right = len(input) - 1
    while left <= right:
        right_cnt = 0
        left_cnt = 0
        while right - right_cnt >= 0 and input[right - right_cnt] != ' ':
            right_cnt += 1

        while left + left_cnt < len(input) and input[left + left_cnt] != ' ':
            left_cnt += 1

        if left_cnt > 0 and right_cnt > 0:
            input[left: left_cnt], input[right - right_cnt + 1: right_cnt] = input[right - right_cnt + 1: right_cnt], input[left: left_cnt]
            right = right - right_cnt
            left = left + left_cnt + 1
        elif left_cnt > 0:
            right -= 1
        elif right_cnt > 0:
            left += 1
        else:
            right -= 1
            left += 1

    return ''.join(input)

def reverse_str_v2(input: str):
    left = 0
    res = ""
    while left < len(input):
        tmp = ''
        while left < len(input) and input[left] != ' ':
            tmp += input[left]
            left += 1
        if tmp != '':
            res = tmp + ' ' + res
        while left < len(input) and input[left] == ' ':
            left += 1
    if res[-1] == ' ':
        return res[0: -1]
    return res






input = "how are   you"


res = reverse_str(input)
print (res)



