class Solution:
    def prodArray(self, nums):
        res = []
      
        for i in range(len(nums)):
            prod = 1                  #reset for every itreation
          
            for j in range(len(nums)):

                if i != j:
                    prod = prod * nums[j]

            res.append(prod)
        return res
