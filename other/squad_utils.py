#!/usr/bin/env python
#-*-coding: utf8-*-

'''
squad utils
@author: plm
@create: 2018-09-01 (Saturday)
@modified: 2018-09-01 (Saturday)
'''
import json


def load_json(file_path):
    with open (file_path, 'r') as f:
        return json.load(f)


def write_json(data_dict, file_path):
    with open (file_path, 'w') as f:
        json.dump(data_dict, f)


def split_one(source_path, target_path):
    '''select one data from source, and write to target'''
    squad = load_json(source_path)
    squad_new = {}
    squad_new['version'] = squad['version']
    squad_new['data'] = []
    squad_new['data'].append(squad['data'][0])
    write_json(squad_new, target_path)


if __name__ == '__main__':
    source_path = "data/source/v1.1/train-v1.1.json"
    target_path = "data/source/v1.1/train-v1.1-less.json"
    split_one(source_path, target_path)
