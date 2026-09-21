arr = [-2, 5, -7, 8, 0, 3, -1]         

# Expected Output:
# Positive: 3
# Negative: 3


positive = 0
negative = 0

for num in arr:
    if num > 0:
        positive += 1
    elif num < 0:
        negative += 1

print("Positive:", positive)
print("Negative:", negative)
