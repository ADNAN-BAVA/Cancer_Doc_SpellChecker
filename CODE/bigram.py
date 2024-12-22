import itertools

from collections import Counter
from nltk.util import ngrams
from preprocess_corpus import word_counter, data
from edit_distance import P, word_candidates

# Bigram-based probabilities
bigrams = Counter(ngrams(data, 2))

def bigram_prob(sent):
    out = P(sent[0].lower())
    for i in range(1, len(sent)):
        out *= bigrams[(sent[i - 1].lower(), sent[i].lower())] / word_counter[sent[i - 1].lower()]
    return out

# Candidate sentences based on bigram probabilities
def sent_all_candidates(sent):
    cands = []
    sent = sent.split()
    word_present = [s.lower() in word_counter for s in sent]
    if all(word_present):
        for i in sent:
            wc = word_candidates(i)[1]
            wc.add(i.lower())
            cands.append(list(wc))
        cands = list(itertools.product(*cands))
    else:
        idx = word_present.index(0)
        words = word_candidates(sent[idx])[1]
        for i in words:
            l = sent.copy()
            l[idx] = i
            cands.append(l)
    return cands

# Find the closest sentence based on bigram probability
def closest_all_sent(sent):
    return ' '.join(max(sent_all_candidates(sent), key=bigram_prob))