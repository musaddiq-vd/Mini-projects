class Solution(object):
    def productExceptSelf(self, nums):

        n = len(nums)

        # Prefix array
        prefix = [1] * n

        for i in range(1, n):
            prefix[i] = prefix[i - 1] * nums[i - 1]

        # Suffix array
        suffix = [1] * n

        for i in range(n - 2, -1, -1):
            suffix[i] = suffix[i + 1] * nums[i + 1]

        # Final answer
        result = [1] * n

        for i in range(n):
            result[i] = prefix[i] * suffix[i]

        return result

