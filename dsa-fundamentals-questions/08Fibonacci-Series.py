
# every next number = sum of previous 2 numbers
# 0 1 1 2 3 5 

n = 3

a = 0 
b = 1

for i in range(n):
    print(a, end=" ")
    a, b = b, a + b
