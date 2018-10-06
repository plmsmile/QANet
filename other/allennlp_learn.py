#!/usr/bin/env python
#-*-coding: utf8-*-

'''
process squad data
@author: plm
@create: 2018-09-01 (Saturday)
@modified: 2018-09-01 (Saturday)
'''
from allennlp.data.dataset_readers.reading_comprehension.squad import SquadReader
from allennlp.data import Vocabulary
from allennlp.data.dataset import Batch
from allennlp.data.token_indexers import TokenCharactersIndexer, SingleIdTokenIndexer
from config import get_config
import allennlp


def read_squad(file_path):
    reader = SquadReader()
    instances = reader.read(file_path)
    vocab = Vocabulary.from_instances(instances)
    #print (len(vocab.get_index_to_token_vocabulary()))

    batch = Batch(instances)
    batch.index_instances(vocab)
    padding_lengths = batch.get_padding_lengths()
    print (padding_lengths)
    tensor_dict = batch.as_tensor_dict(padding_lengths)
    print (tensor_dict.keys())
    print (tensor_dict['passage']['tokens'].shape)
    print (tensor_dict['question']['tokens'].shape)
    print (tensor_dict['span_start'].shape)
    print (tensor_dict['span_end'].shape)
    print (type(tensor_dict['span_end']))
    #batch.print_statistics()
    for instance in instances:
        #print (instance)
        #print (instance.fields['passage'])
        break


def read_squad_word_char(file_path):
    token_indexers = {
            "tokens": SingleIdTokenIndexer(namespace="token_ids"),
            "chars": TokenCharactersIndexer(namespace="token_chars")}
    reader = SquadReader(token_indexers=token_indexers)
    instances = reader.read(file_path)
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


def create_save_vocab(file_path, target_dir, word_min_count, char_min_count):
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


def test_read_squad(opt):
    file_path = opt.train_file
    #read_squad(file_path)
    #read_squad_word_char(file_path)

    target_vocab_dir = opt.vocab_dir
    create_save_vocab(file_path, target_vocab_dir, opt.word_min_count, opt.char_min_count)


if __name__ == '__main__':
    # opt = get_config()
    # test_read_squad(opt)
    print (allennlp.__path__)
