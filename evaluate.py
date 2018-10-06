#!/usr/bin/env python
#-*-coding: utf8-*-

'''
mrc evaluate method, same as the official script. And we can also use allennlp evaluate method.
@author: plm
@create: 2018-10-06
@modified: 2018-10-06
'''

import re
import string
from collections import Counter

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(pred_answer, true_answer):
    '''
    Args
        pred_answer -- str
        true_answer -- str
    Returns:
        f1 --
    '''
    pred_tokens = normalize_answer(pred_answer).split()
    true_tokens = normalize_answer(true_answer).split()
    pred_counter = Counter(pred_tokens)
    true_counter = Counter(true_tokens)
    common = pred_counter & true_counter
    n_same = sum(common.values())
    if n_same == 0:
        return 0
    precision = 1.0 * n_same / len(pred_tokens)
    recall = 1.0 * n_same / len(true_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(pred_answer, true_answer):
    '''exact match
    Args:
        pred_answer -- str
        true_answer -- str
    Returns:
        res -- True:exact match, False: no exact match
    '''
    if normalize_answer(pred_answer) == normalize_answer(true_answer):
        return 1
    return 0


if __name__ == '__main__':
    print ("hello")
