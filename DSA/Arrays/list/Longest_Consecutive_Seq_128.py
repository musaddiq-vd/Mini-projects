class Solution(object):
    def longestConsecutive(self, nums):
        # Convert the list into a set for O(1) average lookup
        num_set = set(nums)
        longest = 0

        # Check each number in the set
        for n in num_set:
            # Start counting only if n is the beginning of a sequence
            if n - 1 not in num_set:
                length = 1

                # Count consecutive numbers
                while n + length in num_set:
                    length += 1

                # Update the longest sequence found so far
                longest = max(longest, length)

        return longest



      
# Short Logic:
# 1. Store all numbers in a set for fast lookup.
# 2. Identify the start of a sequence using (n - 1).
# 3. Count consecutive numbers using the while loop.
# 4. Keep track of the maximum sequence length.
    
      # Time Complexity: O(n) average
      # Space Complexity: O(n)

#Example: [100, 4, 200, 1, 3, 2] → 1,2,3,4 → Answer = 4
