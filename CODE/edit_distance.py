from collections import Counter
import regex as re
from preprocess_corpus import word_counter, data



# Function to calculate edit distance
def editDistance(str1, str2, m, n):
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i][j - 1] + 1, dp[i - 1][j] + 1, dp[i - 1][j - 1] + 1)
    return dp[m][n]

# Word probability based on frequency
def P(word, N=sum(word_counter.values())):
    return word_counter[word.lower()] / N

# Word candidates based on edit distance
def word_candidates(word):
    ed_0 = set()
    ed_1 = set()
    ed_2 = set()
    for w in word_counter:
        ed = editDistance(word.lower(), w, len(word), len(w))
        if ed > 2:
            continue
        elif ed == 0:
            ed_0.add(w)
        elif ed == 1:
            ed_1.add(w)
        elif ed == 2:
            ed_2.add(w)
    #print(f"Candidates for {word}: ed_0={ed_0}, ed_1={ed_1}, ed_2={ed_2}")
    return [ed_0, ed_1, ed_2, {word.lower()}]

# Find the closest word using edit distance
def closest_word(word):
    for i in word_candidates(word):
        if i:
            return max(i, key=P)