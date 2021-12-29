# https://leetcode.com/problems/house-robber/discuss/1070928/Python-DP-bottom-up-bottom-up-O(1)-space-and-top-down-solutions
def rob(nums) -> int:
    max_val = prev = prev_prev = 0

    for i, num in enumerate(nums):
        cur = max(num + prev_prev, prev)
        # if i - 2 >= 0:
        #     cur = max(num + prev_prev, prev)
        # elif i - 1 >= 0:
        #     cur = max(num, prev)
        # else:
        #     cur = num
        max_val = max(max_val, cur)
        prev, prev_prev = cur, prev
    return max_val


# arr = [1, 2, 3, 4, 5]
arr = [1, 1, 1, 1, 1]
result = rob(arr)

print(result)
