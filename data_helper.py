#!/usr/bin/env python
#-*-coding: utf8-*-

'''
squad data
@author: plm
@create: 2018-09-05 (Wednesday)
@modified: 2018-09-05 (Wednesday)
'''
import util
from util import log
from config import get_config, Constant
from allennlp.data import Vocabulary, Instance
from allennlp.data.fields.text_field import TextField
from allennlp.data.dataset import Batch
import torch
import random

PAD = Constant.pad
UNKNOWN = Constant.unk

class SquadData(object):
    '''a simple squad data structure'''

    def __init__(self,
                 passage,
                 question,
                 span_start,
                 span_end,
                 passage_char = None,
                 question_char = None
                 ):
        '''
        Args:
            passage -- [], tokenized words
            question -- [], tokenized words
            span_start -- int, answer start in passage
            span_end -- int, answer end in passage
            passage_char -- [[]], passage word char
            question_char -- [[]], question word char
        '''
        self.passage = passage
        self.question = question
        self.span_start = span_start
        self.span_end = span_end
        self.passage_char = passage_char
        self.question_char = question_char


def get_shared_from_dir_allennlp(vocab_dir):
    '''get word2idx, char2idx by allennlp vocab
    Args:
        vocab_dir -- allennlp vocab saved dir
    Returns:
        word2idx --
        char2idx --
    '''
    word_space = "word2idx"
    char_space = "char2idx"
    vocab = Vocabulary.from_files(vocab_dir)
    word2idx = vocab.get_token_to_index_vocabulary(word_space)
    char2idx = vocab.get_token_to_index_vocabulary(char_space)
    print ("word={}, char={}".format(len(word2idx), len(char2idx)))
    return word2idx, char2idx


def get_tokens_idxs(tokens, token2idx):
    token_idxs = []
    unkown_idx = token2idx.get(Constant.unk)
    for token in tokens:
        idx = token2idx.get(token, unkown_idx)
        token_idxs.append(idx)
    return token_idxs


def read_data(file_path, lower_word=True):
    ''' load [SquadData] from file.
    Args:
        file_path -- pkl file path. made by prepro.py prepro function
        lower_word -- lower all question and passage words. except char tokens
    Returns:
        dataset -- [SquadData]
    '''
    dataset = util.load_data_torch(file_path)
    # lower
    for squad in dataset:
        if lower_word:
            squad.question = [w.lower() for w in squad.question]
            squad.passage = [w.lower() for w in squad.passage]
    return dataset


def convert_dataset_to_idx(dataset, word2idx, char2idx):
    for squad in dataset:
        squad.passage = get_tokens_idxs(squad.passage, word2idx)
        squad.question = get_tokens_idxs(squad.question, word2idx)
        passage_char = [get_tokens_idxs(chs, char2idx) for chs in squad.passage_char]
        question_char = [get_tokens_idxs(chs, char2idx) for chs in squad.question_char]
        squad.passage_char, squad.question_char = passage_char, question_char
    return dataset


def filter_dataset(dataset, max_plen, max_qlen, max_alen, max_wlen):
    '''filter some data of train dataset
    Args:
        dataset -- [SquadData]
        max_plen -- max passage len
        max_qlen -- max question len
        max_alen -- max answer len
        max_wlen -- max word len
    Returns:
        dataset_new -- [SquadData]
    '''
    bad_cnt = 0
    dataset_new = []
    for squad in dataset:
        plen = len(squad.passage)
        qlen = len(squad.question)
        alen = squad.span_end - squad.span_start + 1
        wlen = max([len(w) for w in squad.passage])
        wlen = max(wlen, max([len(w) for w in squad.question]))
        if plen > max_plen or qlen > max_qlen or alen > max_alen or wlen > max_wlen:
            bad_cnt = 0
        dataset_new.append(squad)
    info = "filter: all={}, bad={}, final={}".format(
                    len(dataset), bad_cnt, len(dataset_new))
    log(info)
    return dataset_new


def get_data_loader(dataset, batch_size=1, shuffle=False):
    ''' batch a dataset
    Args:
        dataset -- [data]
        batch_size --
        shuffle -- shuffle the sequnce of all data
    Returns:
        data_loader -- Iterator, get a batch every time
    '''
    if shuffle:
        random.shuffle(dataset)
    start = 0
    end = batch_size
    while (start < len(dataset)):
        batch = dataset[start:end]
        start, end = end, end + batch_size
        yield batch
    if end >= len(dataset) and start < len(dataset):
        batch = dataset[start:]
        yield batch


def _pad_list(dataset, padid=0):
    '''pad a list
    Args:
        dataset -- [[w1, w2]], b*plen
        padid --
    Returns:
        dataset --
        real_lens --
    '''
    max_len = max([len(item) for item in dataset])
    real_lens = []

    for i in range(len(dataset)):
        clen = len(dataset[i])
        real_lens.append(clen)
        gap = max_len- clen
        if gap > 0:
            dataset[i] = dataset[i] + [padid]*gap
    return dataset, real_lens


def _pad_chars(dataset, padid=0):
    ''' pad char data. [sentence], sentence=[word], word=[char]
    Args:
        dataset -- [[[ch1, ch2], [ch1, ch2]]], b*sentence_len*word_len
        padid --
    Returns:
        dataset --
    '''
    # sentence len
    slen = 0
    # word len
    wlen = 0
    for words in dataset:
        slen = max(slen, len(words))
        for word in words:
            wlen = max(wlen, len(word))

    for i in range(len(dataset)):
        # words = dataset[i]
        for j in range(len(dataset[i])):
            # word = words[j]
            cwlen = len(dataset[i][j])
            gap = wlen - cwlen
            if gap > 0:
                dataset[i][j] += [padid] * gap
        cslen = len(dataset[i])
        gap = slen - cslen
        if gap > 0:
            word = [padid] * wlen
            dataset[i] += [word] * gap
    return dataset


def pad_batch_squad(batch, padid=0):
    ''' pad data, and convert to tensors
    Args:
        batch -- [SquadData]
        padid --
    Returns:
        passages --
        questions --
        passage_chars --
        question_chars --
        span_starts --
        span_ends --
    '''
    passages = []
    passage_chars = []
    questions = []
    question_chars = []
    span_starts = []
    span_ends = []

    for squad in batch:
        passages.append(squad.passage)
        questions.append(squad.question)
        passage_chars.append(squad.passage_char)
        question_chars.append(squad.question_char)
        span_starts.append(squad.span_start)
        span_ends.append(squad.span_end)

    passages, passage_lens  = _pad_list(passages, padid)
    questions, question_lens = _pad_list(questions, padid)
    passage_chars = _pad_chars(passage_chars, padid)
    question_chars = _pad_chars(question_chars, padid)

    # to tensor
    passages = torch.LongTensor(passages)
    questions = torch.LongTensor(questions)
    question_chars = torch.LongTensor(question_chars)
    passage_chars = torch.LongTensor(passage_chars)
    span_starts = torch.LongTensor(span_starts)
    span_ends = torch.LongTensor(span_ends)
    return passages, questions, passage_chars, question_chars, span_starts, span_ends


def get_dataset(word2idx, char2idx, serialized_file, opt, lower_word=True):
    '''read serialzed [SquadData] data from a file
    Args:
        word2idx --
        char2idx --
        serialized_file --
        opt --
        lower_word --
    Returns:
        dataset -- [SquadData]. idx value, filtered
    '''
    dataset = read_data(serialized_file, lower_word)
    dataset = filter_dataset(dataset,
                             opt.max_passage_len,
                             opt.max_question_len,
                             opt.max_answer_len,
                             opt.max_word_len)
    dataset = convert_dataset_to_idx(dataset, word2idx, char2idx)
    return dataset


def test_main():
    opt = get_config()
    word2idx = util.read_item2idx_from_file(opt.word2idx_path)
    char2idx = util.read_item2idx_from_file(opt.char2idx_path)
    train_dataset = read_data(opt.train_file, opt.lower_word)
    train_dataset = filter_dataset(train_dataset,
                                   opt.max_passage_len,
                                   opt.max_question_len,
                                   opt.max_answer_len,
                                   opt.max_word_len)
    train_dataset = convert_dataset_to_idx(train_dataset, word2idx, char2idx)
    train_data_loader = get_data_loader(train_dataset, opt.batch_size)
    for batch in train_data_loader:
        res = pad_batch_squad(batch, Constant.padid)
        for tensor in res:
            print (tensor.shape)
        break


if __name__ == '__main__':
    test_main()
