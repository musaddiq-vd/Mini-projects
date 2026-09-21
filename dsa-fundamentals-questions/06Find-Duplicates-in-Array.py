arr = [1, 2, 3, 2, 4, 5, 1, 6] 

# Expected Output:
# Duplicates: 2, 1


seen = set()

for num in arr:
    if num in seen:
        print(num)
    else:
        seen.add(num)
      
