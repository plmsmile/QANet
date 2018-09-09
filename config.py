#!/usr/bin/env python
#-*-coding: utf8-*-

'''

@author: plm
@create: 2018-09-05 (Wednesday)
@modified: 2018-09-05 (Wednesday)
'''
import argparse


def get_config():
    parser = argparse.ArgumentParser()
    # train file
    parser.add_argument("--train_file",
                        type=str,
                        default="data/train/train-less.pkl")
    # word2idx, char2idx
    parser.add_argument("--word2idx_path",
                        type=str,
                        default="data/vocab-less/word2idx.txt")
    parser.add_argument("--char2idx_path",
                        type=str,
                        default="data/vocab-less/char2idx.txt")
    # data lower
    parser.add_argument("--lower_word", action="store_true", default=True)

    # max
    parser.add_argument("--max_passage_len", type=int, default=400)
    parser.add_argument("--max_question_len", type=int, default=50)
    parser.add_argument("--max_answer_len", type=int, default=30)
    parser.add_argument("--max_word_len", type=int, default=16)

    parser.add_argument("--batch_size", type=int, default=2)

    opt = parser.parse_args()
    return opt


class Constant(object):
    pad = "<PADDING>"
    unk = "<UNKNOWN>"
    padid = 0
    unkid = 1



def test():
    opt = get_config()
    print (opt)


if __name__ == '__main__':
    test()


