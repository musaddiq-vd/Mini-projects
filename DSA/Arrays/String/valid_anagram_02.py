from collections import Counter

def isAnagram(s, t):
    counter_s = Counter(s)
    counter_t = Counter(t)

    return counter_s == counter_t
