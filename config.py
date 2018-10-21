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
    parser.register("type", bool, str2bool)

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
    # runrime environment
    runtime = parser.add_argument_group("Environment")
    runtime.add_argument("--batch_size", type=int, default=16)
    runtime.add_argument("--eval_batch_size", type=int, default=16)
    runtime.add_argument("--use_cuda", type=bool, default=False)
    runtime.add_argument("--gpu", type=int, default=0)
    runtime.add_argument("--random_seed", type=int, default=19940620)
    runtime.add_argument("--n_epoch", type=int, default=20)
    # if load model, need this
    runtime.add_argument("--load", type=bool, default=False)
    runtime.add_argument("--load_model", type=str, default="")
    runtime.add_argument("--load_step", type=int, default=0)
    runtime.add_argument("--eval_period", type=int, default=1)

    # files
    files = parser.add_argument_group("FileSystem")
    files.add_argument("--model_name", type=str, default="qanet")
    files.add_argument("--out_dir", type=str, default="out")
    files.add_argument("--run_id", type=str, default="runid")
    files.add_argument("--model_dir", type=str, default="")
    files.add_argument("--eval_log", type=str, default="")


class Constant(object):
    pad = "<PADDING>"
    unk = "<UNKNOWN>"
    padid = 0
    unkid = 1


def test():
    opt = get_config()
    print (opt.Environment)
    print (opt)


if __name__ == '__main__':
    test()


