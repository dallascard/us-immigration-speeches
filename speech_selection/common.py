
def match_tokens(tokens, query_terms):
    """
    Determin if a set of tokens matches a set of query terms
    :param tokens: a list of tokens
    :param query_terms: a set of query terms from query_terms.py
    :return: the set of matches
    """

    tokens = [t.lower() for t in tokens]

    # look for an exact unigram
    overlap = list(set(tokens).intersection(query_terms['exact_unigrams']))
    if len(overlap) > 0:
        return True

    # look for exact bigram matches
    bigrams = [tokens[i-1] + ' ' + tokens[i] for i in range(1, len(tokens))]
    overlap = list(set(bigrams).intersection(query_terms['exact_bigrams']))
    if len(overlap) > 0:
        return True

    # look for a particular 5-gram anywhere
    text = ' '.join(tokens)
    if 'immig' in text.lower():
        return True

    # progressively compare tokens to prefixes
    prefixes = set([t[:11] for t in tokens])
    if len(prefixes.intersection(query_terms['p11'])) > 0:
        return True
    prefixes = set([t[:10] for t in prefixes])
    if len(prefixes.intersection(query_terms['p10'])) > 0:
        return True
    prefixes = set([t[:9] for t in prefixes])
    if len(prefixes.intersection(query_terms['p9'])) > 0:
        return True
    prefixes = set([t[:8] for t in prefixes])
    if len(prefixes.intersection(query_terms['p8'])) > 0:
        return True
    prefixes = set([t[:7] for t in prefixes]) - query_terms['seven_letter_exclude']
    if len(prefixes.intersection(query_terms['p7'])) > 0:
        return True
    prefixes = set([t[:6] for t in prefixes])
    if len(prefixes.intersection(query_terms['p6'])) > 0:
        return True
    prefixes = set([t[:5] for t in prefixes])
    if len(prefixes.intersection(query_terms['p5'])) > 0:
        return True
    prefixes = set([t[:4] for t in prefixes])
    if len(prefixes.intersection(query_terms['p4'])) > 0:
        return True
    prefixes = set([t[:3] for t in prefixes])
    if len(prefixes.intersection(query_terms['p3'])) > 0:
        return True
    prefixes = set([t[:2] for t in prefixes])
    if len(prefixes.intersection(query_terms['p2'])) > 0:
        return True

    return False
