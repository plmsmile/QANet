#!/usr/bin/env python
#-*-coding: utf8-*-

'''
preprocess squad data
@author: plm
@create: 2018-09-01 (Saturday)
@modified: 2018-09-01 (Saturday)
'''
from allennlp.data.dataset_readers.reading_comprehension.squad import SquadReader
from allennlp.data import Vocabulary
from allennlp.data.dataset import Batch
from allennlp.data.token_indexers import TokenCharactersIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from data_helper import SquadData
import argparse
import torch
from util import save_data_torch, load_data_torch
from util import write_item2idx_to_file, read_item2idx_from_file
from config import Constant


def get_config():
    parser = argparse.ArgumentParser()
    # 1. preprocess a squad json
    # raw squad file
    parser.add_argument(
            "--raw_file",
            type=str,
            default="data/source/v1.1/train-v1.1-less.json")
    # target squad file
    parser.add_argument(
            "--target_file",
            type=str,
            default="data/train/train-less.pkl")
    # 2. build a vocab from a squad json
    parser.add_argument("--word_min_count", type=int, default=2)
    parser.add_argument("--char_min_count", type=int, default=2)
    parser.add_argument("--vocab_dir", type=str, default="data/vocab-less")

    opt = parser.parse_args()
    return opt


def build_vocab_allennlp(file_path, target_dir, word_min_count=2, char_min_count=2):
    '''build word2idx, char2idx from a squad file path
    Args:
        file_path --
        target_dir --
        word_min_count --
        char_min_count --
    Returns:
        None
    '''
    namespace_word = "word2idx"
    namespace_char = "char2idx"
    token_indexers = {
            "tokens": SingleIdTokenIndexer(namespace=namespace_word),
            "chars": TokenCharactersIndexer(namespace=namespace_char)}
    min_count = {
            namespace_word: word_min_count,
            namespace_char: char_min_count
            }

    reader = SquadReader(token_indexers=token_indexers)
    instances = reader.read(file_path)

    vocab = Vocabulary.from_instances(instances, min_count=min_count)
    word_cnt = vocab.get_vocab_size(namespace_word)
    char_cnt = vocab.get_vocab_size(namespace_char)
    vocab.save_to_files(target_dir)
    print ("save word2idx={}, char2idx={} to {}".format(
            word_cnt, char_cnt, target_dir))
    word2idx = vocab.get_index_to_token_vocabulary(namespace_word)
    char2idx = vocab.get_index_to_token_vocabulary(namespace_char)
    print (char2idx)
    vocab = Vocabulary.from_files(target_dir)
    char2idx = vocab.get_index_to_token_vocabulary(namespace_char)
    print (char2idx)
    return


def get_squad_from_rawfile(squad_file_path, need_char=False):
    '''
    Args:
        squad_file_path -- a raw squad.json path
        need_char --
    Returns:
        dataset -- [], each item is a SquadData object, which contains
                   passage, question, span_start, span_end. Tokenized
    '''
    namespace_word = "word2idx"
    token_indexers = {
            "tokens": SingleIdTokenIndexer(namespace=namespace_word)}
    reader = SquadReader(token_indexers=token_indexers)
    instances = reader.read(squad_file_path)
    dataset = []
    for instance in instances:
        passage = instance.fields['passage'].tokens
        passage = [word.text for word in passage]
        question = instance.fields['question'].tokens
        question = [word.text for word in question]
        span_start = instance.fields['span_start'].sequence_index
        span_end = instance.fields['span_end'].sequence_index
        item = SquadData(passage, question, span_start, span_end)
        dataset.append(item)
    print ("read {} datas from {}".format(len(dataset), squad_file_path))
    if need_char == True:
        char_tokenizer = CharacterTokenizer()
        # add char tokenizers
        for squad in dataset:
            passage_char = char_tokenizer.batch_tokenize(squad.passage)
            question_char = char_tokenizer.batch_tokenize(squad.question)
            squad.passage_char = [[ch.text for ch in chars] for chars in passage_char]
            squad.question_char = [[ch.text for ch in chars] for chars in question_char]
        print ("add passage_char, question_char for each squad object")
    return dataset


def preproc(squad_source_file, target_file):
    '''generate data.pkl from a raw squad.json'''
    dataset = get_squad_from_rawfile(squad_source_file, True)
    save_data_torch(dataset, target_file)
    dataset_new = load_data_torch(target_file)
    item = dataset_new[0]
    print (item.question_char)
    return


def build_vocab(file_path, target_dir, word_min_count=2, char_min_count=2):
    dataset = get_squad_from_rawfile(file_path, True)
    word2cnt = {}
    lword2cnt = {}
    char2cnt = {}
    for squad in dataset:
        for w in squad.passage + squad.question:
            word2cnt[w] = word2cnt.get(w, 0) + 1
            lword2cnt[w.lower()] = word2cnt.get(w.lower(), 0) + 1
        for word in squad.passage_char + squad.question_char:
            for ch in word:
                char2cnt[ch] = char2cnt.get(ch, 0) + 1

    pad = Constant.pad
    padid = Constant.padid
    unk = Constant.unk
    unkid = Constant.unkid
    word2idx = {pad:padid, unk:unkid}
    lword2idx = {pad:padid, unk:unkid}
    char2idx = {pad:padid, unk:unkid}

    for w, cnt in word2cnt.items():
        if cnt > word_min_count:
            word2idx[w] = len(word2idx)
    for w, cnt in lword2cnt.items():
        if cnt > word_min_count:
            lword2idx[w] = len(lword2idx)
    for ch, cnt in char2cnt.items():
        if cnt > char_min_count:
            char2idx[ch] = len(char2idx)

    write_item2idx_to_file(word2idx, target_dir + "/word2idx_raw.txt")
    write_item2idx_to_file(lword2idx, target_dir + "/word2idx.txt")
    write_item2idx_to_file(char2idx, target_dir + "/char2idx.txt")
    return


def read_squad_allennlp(file_path):
    '''read data, build vocab, batch, padding, to idx
    Args:
        file_path -- raw squad json file
    Returns:
        None
    '''
    token_indexers = {
            "tokens": SingleIdTokenIndexer(namespace="token_ids"),
            "chars": TokenCharactersIndexer(namespace="token_chars")}
    reader = SquadReader(token_indexers=token_indexers)
    instances = reader.read(file_path)
    for instance in instances:
        question = instance.fields['question']
        print (question)
        print (type(question))
        break
    vocab = Vocabulary.from_instances(instances)
    word2idx = vocab.get_index_to_token_vocabulary("token_ids")
    char2idx = vocab.get_index_to_token_vocabulary("token_chars")
    #print (word2idx)
    print (len(word2idx))
    print (len(char2idx))
    print (char2idx)
    batch = Batch(instances)
    batch.index_instances(vocab)
    padding_lengths = batch.get_padding_lengths()
    print (padding_lengths)
    tensor_dict = batch.as_tensor_dict(padding_lengths)
    print (tensor_dict['passage']['tokens'].shape)
    print (tensor_dict['passage']['chars'].shape)
    print (tensor_dict['question']['tokens'].shape)
    print (tensor_dict['question']['chars'].shape)
    print (tensor_dict['span_start'].shape)
    print (tensor_dict['span_end'].shape)


if __name__ == '__main__':
    opt = get_config()
    preproc(opt.raw_file, opt.target_file)
    #build_vocab(opt.raw_file, opt.vocab_dir, opt.word_min_count, opt.char_min_count)
    #read_squad_allennlp(opt.raw_file)
