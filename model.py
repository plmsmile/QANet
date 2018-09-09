#!/usr/bin/env python
#-*-coding: utf8-*-

'''
QANet model
@author: plm
@create: 2018-09-08 (Saturday)
@modified: 2018-09-08 (Saturday)
'''
import torch
import torch.nn as nn


# for test
from config import get_config
from data import get_shared_from_dir
import numpy as np


class QANet(nn.Module):

    def __init__(self):
        super().__init__()


def test_main():
    opt = get_config()
    word2idx, char2idx = get_shared_from_dir(opt.vocab_dir)


if __name__ == '__main__':
    test_main()


