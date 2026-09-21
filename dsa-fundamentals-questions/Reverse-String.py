# Sample Input: "hello"         Expected Output: "olleh"


# 1 reverse loop
for i in range(len(st)-1, -1, -1):
    print(st[i], end="")



# 2 swapping using two pointer
st = "hello"
st = list(st)

l = 0
r = len(st) - 1

while l < r:
    st[l], st[r] = st[r], st[l]
    l += 1
    r -= 1

st = "".join(st)
print(st)

# note : string is immutable hence convert into list
