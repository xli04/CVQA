import nltk
from nltk.corpus import wordnet as wn

import nltk

def word_tokenize_safe(text):
    # Directly load the pickled tokenizer (no punkt_tab involved)
    with open('/root/nltk_data/tokenizers/punkt/english.pickle', 'rb') as f:
        punkt_tokenizer = nltk.data.pickle.load(f)
    return punkt_tokenizer.tokenize(text)


def wup(w1, w2, alpha):
    """
    Basic Wu-Palmer similarity using WordNet
    """
    synsets1 = wn.synsets(w1)
    synsets2 = wn.synsets(w2)

    if not synsets1 or not synsets2:
        return 0

    max_score = max(
        (s1.wup_similarity(s2) or 0)
        for s1 in synsets1 for s2 in synsets2
    )

    return max_score if max_score >= alpha else max_score * 0.1

def wups(words1, words2, alpha):
    sim = 1.0
    flag = False
    for w1 in words1:
        max_sim = 0
        for w2 in words2:
            word_sim = wup(w1, w2, alpha)
            max_sim = max(max_sim, word_sim)
        if max_sim == 0:
            continue
        sim *= max_sim
        flag = True
    return sim if flag else 0.0

def get_wups(pred, truth, alpha):
    pred = word_tokenize_safe(pred)
    truth = word_tokenize_safe(truth)
    return min(wups(pred, truth, alpha), wups(truth, pred, alpha))

