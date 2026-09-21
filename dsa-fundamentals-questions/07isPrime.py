# prime nums eg : 2, 3, 5, 7, 11, 13 
# numbers that can only divisible by 1 and itself, it should have only 2 factors


n = 13
def isprime(n):
    if n < 2:
        return "not prime"
    
    for i in range(2, int(n ** 0.5) +1):
        if n % i == 0:
            return "not prime"
        
    return "prime"

print(isprime(n))
