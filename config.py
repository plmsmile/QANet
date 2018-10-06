#!/usr/bin/env python
#-*-coding: utf8-*-

'''

@author: plm
@create: 2018-09-05 (Wednesday)
@modified: 2018-09-05 (Wednesday)
'''
import argparse


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def get_config():
    parser = argparse.ArgumentParser()
    parser.register("type", 'bool', str2bool)

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

    # EncoderBlock's input and output size
    parser.add_argument("--encode_size", type=int, default=128)

    # dropout probability
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dropout_char", type=float, default=0.05)

    # optimizer
    parser.add_argument("--n_warmup_steps", type=int, default=4000)


    add_train_args(parser)
    opt = parser.parse_args()
    return opt


def add_train_args(parser):
    runtime = parser.add_argument_group("Envioronment")

    runtime.add_argument("--batch_size", type=int, default=16)
    runtime.add_argument("--use_cuda", type="bool", default=True)
    runtime.add_argument("--gpu", type=int, default=1)
    runtime.add_argument("--random_seed", type=int, default=19940620)



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


