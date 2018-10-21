#!/usr/bin/env python
#-*-coding: utf8-*-

'''
some util method
@author: plm
@create: 2018-09-05 (Wednesday)
@modified: 2018-09-05 (Wednesday)
'''

import torch
import numpy as np
import random
import time


def save_data_torch(data_object, target_file_path):
    '''serialize a object to a file'''
    torch.save(data_object, target_file_path)
    # log("torch save {} datas to {}".format(len(data_object), target_file_path))
    return


def load_data_torch(data_file_path):
    '''deserialize a object from a file'''
    data_object = torch.load(data_file_path)
    # log("torch load {} datas from {}".format(len(data_object), data_file_path))
    return data_object


def log(info):
    print ("[INFO] {}".format(info))


def write_list_to_file(lines, target_file, append=False):
    '''write some standard lines to a file
    Args:
        lines --
        target_file --
        append -- full-write or append-write
    Returns:
        None
    '''
    mode = 'a' if append else 'w'
    with open(target_file, mode) as f:
        for line in lines:
            f.write(line + "\n")
    log("write {} lines to {}".format(len(lines), target_file))
    return


def read_list_from_file(source_file):
    '''read list from a file'''
    lines = []
    with open(source_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line == "" or len(line) <= 0:
                continue
            lines.append(line)
    log("read {} lines from {}".format(len(lines), source_file))
    return lines


def write_dict_to_file(d, target_file, split_char=" ", append=False):
    '''write a simple dict to a file with lines. k v, k v
    Args:
        d -- dict, key=str, value=[str, int, float]
        target_file --
        split_char --
        append -- full-write or append-write
    Returns:
        None
    '''
    lines = []
    for k, v in d.items():
        line = k + split_char + str(v)
        lines.append(line)
    write_list_to_file(lines, target_file, append)
    return


def read_dict_from_file(src_file, split_char=" ", vtype=str):
    '''read a dict from a file
    Args:
        src_file --
        split_char --
        vtype -- value type, [str, int, float]
    Returns:
        d -- the dict
    '''
    lines = read_list_from_file(src_file)
    d = {}
    for line in lines:
        k, v = line.split(split_char)
        if k is None or v is None or v == "" or len(v) <= 0:
            continue
        v = vtype(v)
        d[k] = v
    return  d


def write_item2idx_to_file(item2idx, target_file, split_char=" "):
    ''' write item2idx to a file. item idx.
    Args:
        item2idx --
        target_file --
        split_char -- a split char between item and idx
    Returns:
        None
    '''
    item2idx_list = sorted(item2idx.items(), key=lambda d: d[1])
    lines = []
    for item, cnt in item2idx_list:
        lines.append (item + split_char + str(cnt))
    write_list_to_file(lines, target_file)
    return


def read_item2idx_from_file(item2idx_file, split_char=" "):
    '''load a item2idx dict from a file
    Args:
        item2idx_file --
        split_char --
    Returns:
        item2idx -- dict
    '''
    return read_dict_from_file(item2idx_file, split_char, int)


def get_span_from_probs(prob_start, prob_end, topn=1, maxlen=None):
    '''take argmax of constrained prob_start*prob_end
    Args:
        prob_start -- [b, slen]
        prob_end -- [b, slen]
        topn -- best topn span
        maxlen -- max span length to consider
    Returns:
        pred_span -- [spans], len=batch_size, spans=[(start, end)], len=topn
        pred_score -- [scores], len=batch_size, scores=[score], len=topn
    '''
    batch_size, slen = prob_start.size()
    pred_span = []
    pred_score = []
    maxlen = maxlen or slen
    for i in range(batch_size):
        # [slen, slen]
        scores = torch.ger(prob_start[i], prob_end[i])
        # keep upper triangular start<=end
        scores = scores.triu()
        # keep lower triangular end-start+1<=maxlen
        scores = scores.tril(maxlen - 1)
        scores_flat = scores.detach().cpu().numpy().flatten()

        # score topn's index, desc sort
        idx_sort = None
        if topn == 1:
            # top 1
            idx = np.argmax(scores_flat)
            idx_sort = [idx]
        elif slen < topn:
            # top slen
            idx_sort = np.argsort(scores_flat)
        else:
            # topn
            # idxs = np.argpartition(scores_flat, -topn)[-topn:]
            idxs = np.argpartition(-scores_flat, topn)[0:topn]
            idx_sort = idxs[np.argsort(-scores_flat[idxs])]
        start_idx, end_idx = np.unravel_index(idx_sort, scores.shape)
        start_end = list(zip(start_idx, end_idx))
        scores_top = scores_flat[idx_sort]
        pred_span.append(start_end)
        pred_score.append(scores_top)
    return pred_span, pred_score


def set_random_seed(seed, cuda=False):
    '''set random seed'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total


if __name__ == '__main__':
    print ("hello")
