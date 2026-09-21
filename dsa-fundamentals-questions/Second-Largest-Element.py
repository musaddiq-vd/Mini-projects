# arr = [10, 5, 8, 10, 3, 7]            Expected:  Second Largest: 8


arr = [10, 5, 8, 10, 3, 7]

largest = arr[0]
slargest = float('-inf')   # negative infinity.

for i in range(1, len(arr)):
    if arr[i] > largest:
        slargest = largest
        largest = arr[i]

    elif arr[i] > slargest and arr[i] < largest:
        slargest = arr[i]

print(slargest)
