
# find first character with frequncy  == 1  
# input s = "leetcode"      1output:  0     coz l at 0th index
# input s = "loveleetcode"  2output:  2     coz v at 2nd index
 
def firstchar(s):
    count = {}
    for ch in s:
        count[ch] = count.get(ch, 0) + 1

    for i in range(len(s)):
        if count[s[i]] == 1:
            return i

print(firstchar(s))
