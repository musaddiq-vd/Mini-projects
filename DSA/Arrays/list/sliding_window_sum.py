def sw(arr, k):
    # Calculate the sum of the first k elements
    curr_sum = sum(arr[:k])
    max_sum = curr_sum

    # Start from k because the first k elements are already processed
    for i in range(k, len(arr)):

        # Add the new element to the current window
        curr_sum += arr[i]

        # Remove the old/left element from the window
        curr_sum -= arr[i - k]
      
        max_sum = max(max_sum, curr_sum)

    return max_sum
