#!/usr/bin/env python
#-*-coding: utf8-*-

'''

@author: plm
@create: 2018-09-27 (星期四)
@modified: 2018-09-27 (星期四)
'''
from allennlp.common.squad_eval import normalize_answer
import torch
import torch.optim as optim
import torch.nn as nn
from layers import DepthwiseSeparableConv
from util import load_data_torch

def test():
    s = "i love a the you."
    print ("raw=[{}]".format(s))
    print (normalize_answer(s))


class Model(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.n1 = nn.Linear(input_size, input_size)
        self.n2 = nn.Linear(input_size, input_size)
        self.n3 = nn.Linear(input_size, 1)

    def forward(self, inputs):
        x = self.n1(inputs)
        x = self.n2(x)
        preds = self.n3(x).squeeze()
        return preds


def test_nn():
    batch_size = 6
    input_size = 10
    epoch = 12

    # network
    model = Model(input_size)

    # loss
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # data
    inputs = torch.randn(batch_size, input_size)
    labels = torch.randn(batch_size)

    # forward
    for i in range(epoch):
        preds = model(inputs)
        loss = loss_func(preds, labels)
        print ("i={}, loss={}".format(i, round(loss.item(), 3)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_conv():
    # conv = DepthwiseSeparableConv(100, 100, 5, False)
    input_embedding = load_data_torch("input_embedding.pkl")
    conv2d = input_embedding.conv2d
    inputs = load_data_torch("inputs.pkl")
    conv2d(inputs, True)
    for param in conv2d.parameters():
        print (param)







if __name__ == '__main__':
    print ("hello")
    # test()
    # test_nn()
    test_conv()


