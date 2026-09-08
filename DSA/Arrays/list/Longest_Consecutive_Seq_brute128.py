class Solution:
    def longestConsecutive(self, nums):
        #if empty
        if not nums:
            return 0

        nums.sort()

        longest = 1
        current = 1

        for i in range(1, len(nums)):

            # Consecutive number
            if nums[i] == nums[i - 1] + 1:
                current += 1

            # New/different number (sequence breaks)
            elif nums[i] != nums[i - 1]:
                current = 1

            longest = max(longest, current)

        return longest
