class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:

        count = {}
        # Buckets: index = frequency
        buckets = [[] for _ in range(len(nums) + 1)]

        # Frequency count
        for num in nums:
            count[num] = count.get(num, 0) + 1

        # Put each number into its frequency bucket
        for num, freq in count.items():
            buckets[freq].append(num)

        result = []
        # Start from highest frequency
        for freq in range(len(buckets) - 1, 0, -1):

            for num in buckets[freq]:
                result.append(num)

                # Stop when we have k elements
                if len(result) == k:
                    return result
                  
#Time Complexity: O(n)
#Space Complexity: O(n)
