def isAnagram(s, t):
    # Different lengths cannot be anagrams
    if len(s) != len(t):
        return False

    s_dict = {}
    t_dict = {}

    # Count frequency of each character in both strings
    for i in range(len(s)):
        s_dict[s[i]] = s_dict.get(s[i], 0) + 1
        t_dict[t[i]] = t_dict.get(t[i], 0) + 1

    # Same character frequencies = anagram
    return s_dict == t_dict

#Length check → character frequency count → both dictionaries compare → same means Anagram.
