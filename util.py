#!/usr/bin/env python
#-*-coding: utf8-*-

'''
some data utils
@author: plm
@create: 2018-09-05 (Wednesday)
@modified: 2018-09-05 (Wednesday)
'''

import torch


def save_data_torch(data_object, target_file_path):
    '''serialize a object to a file'''
    torch.save(data_object, target_file_path)
    log("torch save {} datas to {}".format(len(data_object), target_file_path))
    return


def load_data_torch(data_file_path):
    '''deserialize a object from a file'''
    data_object = torch.load(data_file_path)
    log("torch load {} datas from {}".format(len(data_object), data_file_path))
    return data_object


def log(info):
    print ("[INFO] {}".format(info))


def write_list_to_file(lines, target_file):
    '''write some standard lines to a file
    Args:
        lines --
        target_file --
    Returns:
        None
    '''
    with open(target_file, "w") as f:
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
    lines = read_list_from_file(item2idx_file)
    item2idx = {}
    for line in lines:
        item, cnt = line.split(split_char)
        if item is None or cnt is None or item == "" or len(cnt) <= 0:
            continue
        cnt = int(cnt)
        item2idx[item] = cnt
    return item2idx


if __name__ == '__main__':
    print ("hello")
