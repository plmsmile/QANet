#!/usr/bin/env python
#-*-coding: utf8-*-

'''
some numpy methods
@author: plm
@create: 2018-09-28 (星期五)
@modified: 2018-09-28 (星期五)
'''

import numpy as np


def test_argpartition():
    '''partition sort. small, x, big'''
    # 1. one index
    a = np.array([9, 4, 3, 0, 5, 6, 1])
    print (a)
    # left--less than 4, middle--4, right--bigger than 4. sort asc
    i3 = np.argpartition(a, 4)
    print(i3, a[i3])
    # two index. index 3-4 num are in their final positions
    index = np.argpartition(a, (3, 4))
    print (index, a[index])

    # 2. topn
    # 从小到大排列，倒数第5个数排好（最大的第5个数）
    # idx = np.argpartition(a, -5)[-5:]
    idx = np.argpartition(-a, 5)[0:5]
    # no order
    nums = a[idx]
    # sort, order, desc sort. -nums
    idx = np.argsort(-nums)
    topn = nums[idx]
    print (topn)


def test_unravel_index():
    ''' flatten -- unravel_index
    Args:
        indices --
        dims -- the shape
    Returns:
        dim1_idx --
        dim2_idx --
        ...
    '''
    flatten_indices = [2, 3, 4]
    shape = (3, 3)
    dim1_idx, dim2_idx = np.unravel_index(flatten_indices, shape)
    print (dim1_idx, dim2_idx)


if __name__ == '__main__':
    print ("hello")
    # test_argpartition()
    test_unravel_index()



