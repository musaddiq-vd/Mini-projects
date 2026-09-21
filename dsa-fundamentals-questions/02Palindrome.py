# Sample Input:  st = "madam"             Expected Output: Palindrome 
# Another Input:  st = "hello"            Expected Output: Not Palindrome


def pal(st):

    l = 0
    r = len(st) - 1

    while l < r:

        if st[l] != st[r]:
            return False
        l+=1
        r-=1

    return True
