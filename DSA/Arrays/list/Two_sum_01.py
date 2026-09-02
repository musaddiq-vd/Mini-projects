class Solution:
    def twoSum(self, nums, target):
        seen = {}

        for i, num in enumerate(nums):
            complement = target - num

            if complement in seen:
                return [seen[complement], i]

            seen[num] = i

        return []

"""
seen → Stores each number with its index.

complement = target - num → Finds the required second number.

If complement is already in seen → The pair is found.

return [seen[complement], i] → Returns both indices.

Time: O(n)
Space: O(n)

"""
