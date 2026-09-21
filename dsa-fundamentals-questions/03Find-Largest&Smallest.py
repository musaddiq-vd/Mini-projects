arr = [5, 2, 9, 1, 7]

# Expected Output:
# Largest: 9
# Smallest: 1

# 1. using in built function
print(max(arr))
print(min(arr))


# 2. custome
def maximum(arr):
    maxm = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > maxm:
            maxm = arr[i]
    return maxm


def minimum(arr):
    mini = arr[0]
    for i in range(1, len(arr)):
        if arr[i] < mini:
            mini = arr[i]
    return mini


print("Maximum:", maximum(arr))
print("Minimum:", minimum(arr))
