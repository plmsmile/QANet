#!/usr/bin/env python
#-*-coding: utf8-*-

'''
run for QANet
@author: plm
@create: 2018-09-23
@modified: 2018-09-23
'''
import torch
from model import QANet
from config import get_config, Constant
import numpy as np
import util
from data_helper import get_dataset_qid2info, get_data_loader, pad_batch_squad
from optim import ScheduledOptimizer
import torch.optim as optim
import random
import torch.nn.functional as F

import string
import re
from collections import Counter


class MRCModel(object):

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.global_step = 0

    def save(self, path=None):
        pass

    def load(self):
        pass

    def train(self, batch):
        pass

    def init_optimizer(self, state_dict=None):
        '''init a ScheduledOptimizer object
        Args:
            state_dict -- state_dict, may load from file
        Returns:
            None
        '''
        optimizer = optim.Adam(
                        filter(lambda param: param.requires_grad, self.model.parameters()),
                        betas=(0.8, 0.999), eps=1e-07)
        if state_dict is not None:
            optimizer.load_state_dict(state_dict)
        optimizer = ScheduledOptimizer(optimizer, opt.encode_size, opt.n_warmup_steps)
        self.optimizer = optimizer
        return


def set_random_seed(seed, cuda=False):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def prepare_embed_mat(word2idx, char2idx):
    word_embed_size = 300
    char_embed_size = 200
    word_mat = [np.random.normal(0, 1, word_embed_size) for i in range(len(word2idx))]
    char_mat = [np.random.normal(0, 1, char_embed_size) for i in range(len(char2idx))]
    word_mat = torch.FloatTensor(word_mat)
    char_mat = torch.FloatTensor(char_mat)
    return word_mat, char_mat


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def cal_loss(prob_starts, prob_ends, starts, ends):
    ''' calculate a batch's loss
    Args:
        prob_starts -- softmax probs, [b, plen]
        prob_ends -- softmax probs, [b, plen]
        starts -- [b]
        ends -- [b]
    Returns:
        loss -- avg loss
    '''
    prob_starts, prob_ends = torch.log(prob_starts), torch.log(prob_ends)
    loss_start = F.nll_loss(prob_starts, starts)
    loss_end = F.nll_loss(prob_ends, ends)
    loss = (loss_start + loss_end) / 2
    return loss


def cal_performance(prob_starts, prob_ends, qids, qid2info):
    '''calculate f1 score
    Args:
        prob_starts --
        prob_ends --
        qids -- [] question id
        qid2info -- {}, question_id and some info
    Returns:
        f1 -- f1 score
        em -- exact match score
    '''
    pred_span, pred_score = decode(prob_starts, prob_ends)
    f1 = 0
    em = 0
    total = 0
    for i, qid in enumerate(qids):
        info = qid2info[qid]
        span = pred_span[i][0]
        widx_start, widx_end = span[0], span[1]
        chidx_start = info['passage_token_offsets'][widx_start][0]
        # not conclude the chidx_end char
        chidx_end = info['passage_token_offsets'][widx_end][1]
        answer_pred = info['passage'][chidx_start:chidx_end]
        f1 += f1_score(answer_pred, info['answer'])
        em += exact_match_score(answer_pred, info['answer'])
        total += 1
    f1 = 100.0 * f1 / total
    em = 100.0 * em / total
    return round(f1, 2), round(em, 2)


def f1_score(pred_answer, true_answer):
    '''
    Args
        pred_answer -- str
        true_answer -- str
    Returns:
        f1 --
    '''
    pred_tokens = normalize_answer(pred_answer).split()
    true_tokens = normalize_answer(true_answer).split()
    pred_counter = Counter(pred_tokens)
    true_counter = Counter(true_tokens)
    common = pred_counter & true_counter
    n_same = sum(common.values())
    if n_same == 0:
        return 0
    precision = 1.0 * n_same / len(pred_tokens)
    recall = 1.0 * n_same / len(true_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(pred_answer, true_answer):
    '''exact match
    Args:
        pred_answer -- str
        true_answer -- str
    Returns:
        res -- True:exact match, False: no exact match
    '''
    if normalize_answer(pred_answer) == normalize_answer(true_answer):
        return 1
    return 0



def decode(prob_start, prob_end, topn=1, maxlen=None):
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


def go_train(model, train_dataset, train_qid2info, opt):
    params = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = optim.Adam(
                    filter(lambda param: param.requires_grad, model.parameters()),
                    betas=(0.8, 0.999), eps=1e-07)
    optimizer = ScheduledOptimizer(optimizer, opt.encode_size, opt.n_warmup_steps)

    device = opt.device
    global_step = 0
    for e in range(500):
        print ("--------epoch={}----------".format(e+1))
        train_data_loader = get_data_loader(train_dataset, opt.batch_size, True)
        for batch in train_data_loader:
            global_step += 1
            res = pad_batch_squad(batch, Constant.padid)
            tensors, qids = res[:-1], res[-1]
            tensors = [tensor.to(device) for tensor in tensors]
            passages, questions, passage_chars, question_chars, span_starts, span_ends = tensors
            prob_start, prob_end = \
                        model(passages, questions, passage_chars, question_chars)
            loss = cal_loss(prob_start, prob_end, span_starts, span_ends)
            f1, em = cal_performance(prob_start, prob_end, qids, train_qid2info)
            print ("{} loss={}, f1={}, em={}".format(global_step, round(loss.item(), 2), f1, em))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10)
            optimizer.step_and_update_lr()


def show_bad(squad_list):
    cnt = 0
    for squad in squad_list:
        l1 = len(squad.question)
        l2 = len(squad.question_char)
        if l1 != l2:
            # print ("q={}, qch={}".format(l1, l2))
            cnt += 1
    print ("bad = {}, all={}".format(cnt, len(squad_list)))


def get_model(opt, word_mat, char_mat):
    model = QANet(word_mat, char_mat,
                  opt.dropout, opt.dropout_char, opt.max_passage_len,
                  opt.encode_size)

    return model


def test_main():
    opt = get_config()
    if opt.use_cuda and torch.cuda.is_available():
        opt.use_cuda = True
        opt.device = torch.device("cuda:"+str(opt.gpu))
    else:
        opt.use_cuda = False
        opt.device = torch.device("cpu")
    set_random_seed(opt.random_seed, opt.use_cuda)
    print (opt)
    word2idx = util.read_item2idx_from_file(opt.word2idx_path)
    char2idx = util.read_item2idx_from_file(opt.char2idx_path)
    word_mat, char_mat = prepare_embed_mat(word2idx, char2idx)
    train_dataset, train_qid2info = \
        get_dataset_qid2info(word2idx, char2idx, opt.train_file, opt, opt.lower_word)
    model = get_model(opt, word_mat, char_mat)
    if opt.use_cuda:
        model = model.to(device=opt.device)
    go_train(model, train_dataset, train_qid2info, opt)


def test_decode():
    b = 2
    slen = 5
    prob_start = F.softmax(torch.randn(b, slen), -1)
    prob_end = F.softmax(torch.randn(b, slen), -1)
    pred_span, pred_score = decode(prob_start, prob_end, maxlen=4)
    print (pred_span[0])


if __name__ == '__main__':
    print ("hello")
    test_main()
