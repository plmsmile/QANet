#!/usr/bin/env python
#-*-coding: utf8-*-

'''
some pytorch api test
@author: plm
@create: 2018-09-10 (Monday)
@modified: 2018-09-10 (Monday)
'''
import torch
import torch.nn as nn


data = [[1, 2, 3], [4, 5, 6]]


def get_tensor():
    return torch.Tensor(data)


def test_conv1d():
    '''conv1d'''
    in_channels = 8
    out_channels = 16
    kernal_size = 3
    # default value
    stride = 1
    padding = 0

    conv = nn.Conv1d(in_channels, out_channels, kernal_size, stride, padding)
    # batch, in_channels, in_seqlen
    in_seqlen = 10
    inputs = torch.randn(2, in_channels, in_seqlen)
    outputs = conv(inputs)
    # 2,16,8
    print (outputs.shape)


def test_conv1d_depthwise():
    '''depthwise conv1d. groups=in_channels'''
    in_channels = 8
    out_channels = 1 * in_channels
    kernal_size = 3
    # default value
    stride = 1
    padding = 0
    # out_seqlen == in_seqlen
    padding = kernal_size // 2
    conv = nn.Conv1d(in_channels, out_channels, kernal_size, stride, padding,
                     groups=in_channels)
    # batch, in_channels, in_seqlen
    in_seqlen = 10
    inputs = torch.randn(2, in_channels, in_seqlen)
    outputs = conv(inputs)
    # 2,8,8
    print (inputs.shape)
    print (outputs.shape)


def test_permute():
    '''exchange dimensions of a tensor'''
    a = torch.randn(3, 4, 5)
    print (a.permute(2, 0, 1).shape)


def test_max():
    a = torch.randn(2, 3)
    print (a)
    # the max of all values. a single value
    torch.max(a)
    # dim
    values, idxs = torch.max(a, dim=0)
    print (values)

    # argmax
    idxs = torch.argmax(a, dim=1)
    print (idxs)


def test_layer_norm():
    b = 2
    input_size = 5
    x = torch.randint(10, (b,input_size))
    print (x)
    layer_norm = nn.LayerNorm(input_size)
    x = layer_norm(x)
    print (x)


def test_bmm():
    '''matrix multiply'''
    batch_size = 10
    a = torch.randn(batch_size, 3, 4)
    b = torch.randn(batch_size, 4, 5)
    # [b, 3, 5]
    c = torch.bmm(a, b)
    print (c.shape)


def test_eq():
    a = torch.Tensor([[0, 1], [2, 3]])
    print (a.eq(0))


def test_unsqueeze():
    '''add a dim'''
    # [2, 3]
    a = torch.Tensor(data)
    # add a dim [2, 1, 3]
    b = a.unsqueeze(1)
    print (b.shape)
    # remove a dim. [2, 3]
    c = a.squeeze(1)
    print (c.shape)


def test_expand():
    # expand the size=1 's dim
    # [3, 1]
    a = torch.Tensor([[1], [2], [3]])
    print (a.shape)
    # [3, 4]
    b = a.expand(3, 4)
    print (b)
    print (b.shape)
    # print (a.expand(-1, 4).shape)


def test_unsqueeze_expand():
    # [2, 3]
    a = torch.Tensor(data)
    a = a.eq(1)
    # [2, 1, 3]
    a = a.unsqueeze(1)
    print (a.shape)
    n = 4
    a = a.expand(-1, n, -1)
    print (a.shape)


def test_masked_fill():
    a = torch.Tensor(data)
    mask = a.eq(1)
    b = a.masked_fill(mask, 0)
    print (b)


def test_dot():
    # [2, 3], [batch_size, seq_len]
    a = torch.Tensor(data).ne(1).unsqueeze(-1).type(torch.float)
    print (a)
    #print (a.shape)
    b = torch.randn(2, 3, 4)
    print (b)
    # b = torch.Tensor(data)
    c = a * b
    print (c)
    #print (b * a)


def test_mul():
    '''element wise'''
    a = torch.Tensor(data)
    b = torch.Tensor(data)
    print (a * b )
    print (a.mul(b))


def test_transpose_permute():
    a = torch.Tensor(data)
    print (a.transpose(0, 1))
    print (a.permute(1, 0))


def test_matrix_mask():
    # [1, 3]
    a = torch.Tensor([1, 1, 0]).unsqueeze(0)

    # 1. multiply method
    # [1, 3, 1]
    b = a.unsqueeze(2)
    # [1, 3, 3]
    res = b.bmm(b.transpose(1, 2))
    print (res)

    # 2. expand method
    res = a.unsqueeze(1).expand(-1, 3, -1)
    print (res)


def test_logsoftmax():
    func = nn.LogSoftmax()
    x = torch.randn(2, 3)
    print (func(x))


def test_l1loss():
    loss = nn.L1Loss(reduction="elementwise_mean")
    # input = torch.randn(3, 5, requires_grad=True)
    input = torch.randn(3, 5, requires_grad=False)
    target = torch.randn(3, 5)
    output = loss(input, target)
    print (output)
    # output.backward()


def test_ger():
    '''two vector's product, n, m, return n*m'''
    v1 = torch.Tensor([1, 2])
    v2 = torch.Tensor([1, 2, 3])
    # [2, 3]
    res = torch.ger(v1, v2)
    print (res)


def test_tril():
    '''keep the lower triangular part
    Args:
        input -- 2-D tensor
        dagonal -- 0 -- default, on and below the main dagonal
        dagonal -- 1 -- main dagonal adds 1, move up
        dagonal -- 2 -- main dagonal adds 2, move up
        dagonal -- -1 -- main dagonal sub 1, move down
        dagonal -- -2 -- main dagonal sub 2, move down
    '''
    a = torch.randn(6, 6)
    print ("---raw----")
    print (a)
    print ("---tril 0----")
    print (a.tril())
    print ("---tril 3----")
    print (a.tril(3))
    # print (b)
    # print (a.tril(1))
    # print (a.tril(2))
    # print (a.tril(-1))
    # print (a.tril(-2))


def test_triu():
    '''keep up the upper triangular part'''
    a = torch.randn(4, 4)
    b = a.triu()
    print (b)
    print (b.triu(1))


if __name__ == '__main__':
    print ("hello")
    # test_conv1d()
    # test_conv1d_depthwise()
    # test_permute()
    # test_max()
    # test_layer_norm()
    # test_bmm()
    # test_eq()
    # test_unsqueeze()
    # test_expand()
    # test_unsqueeze_expand()
    # test_masked_fill()
    # test_dot()
    # test_mul()
    # test_transpose_permute()
    # test_matrix_mask()
    # test_logsoftmax()
    # test_l1loss()
    # test_ger()
    test_tril()
    # test_triu()
