#!/usr/bin/env python
#-*-coding: utf8-*-

'''
run for QANet
@author: plm
@create: 2018-09-23 (星期日)
@modified: 2018-09-23 (星期日)
'''
import torch
from model import QANet
from config import get_config, Constant
import numpy as np
import util
from data_helper import get_dataset, get_data_loader, pad_batch_squad
import random


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def _prepare_embed_mat(word2idx, char2idx):
    word_embed_size = 100
    char_embed_size = 100
    word_mat = [np.random.normal(0, 1, word_embed_size) for i in range(len(word2idx))]
    char_mat = [np.random.normal(0, 1, char_embed_size) for i in range(len(char2idx))]
    word_mat = torch.FloatTensor(word_mat)
    char_mat = torch.FloatTensor(char_mat)
    return word_mat, char_mat


def go_train(model, train_dataset, opt):
    train_data_loader = get_data_loader(train_dataset, opt.batch_size, True)

    for batch in train_data_loader:
        passages, questions, passage_chars, question_chars, \
                span_starts, span_ends = pad_batch_squad(batch, Constant.padid)
        prob_start, prob_end = model(passages, questions, passage_chars, question_chars)
        print ("------prob start------", prob_start.shape)
        break


def get_model(opt, word_mat, char_mat):
    model = QANet(word_mat, char_mat,
                  opt.dropout, opt.dropout_char, opt.max_passage_len,
                  opt.encode_size)

    return model


def test_main():
    set_random_seed(19940620)
    opt = get_config()
    print (opt)
    word2idx = util.read_item2idx_from_file(opt.word2idx_path)
    char2idx = util.read_item2idx_from_file(opt.char2idx_path)
    word_mat, char_mat = _prepare_embed_mat(word2idx, char2idx)
    train_dataset = get_dataset(word2idx, char2idx, opt.train_file, opt, opt.lower_word)
    model = get_model(opt, word_mat, char_mat)
    go_train(model, train_dataset, opt)


if __name__ == '__main__':
    print ("hello")
    test_main()
