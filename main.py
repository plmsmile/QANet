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
from data_helper import get_dataset_qid2info, get_data_loader
from optim import ScheduledOptimizer
import torch.optim as optim
import random
import torch.nn.functional as F

from util import get_span_from_probs, set_random_seed
from evaluate import f1_score, exact_match_score

import logging

logger = logging.getLogger(__name__)


class MRCManager(object):

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.global_step = 0
        self.params = filter(lambda param: param.requires_grad, self.model.parameters())

    def save(self):
        saved_params = {
                'state_dict':self.model.state_dict(),
                'global_step':self.global_step,
                'optimizer': self.optimizer._optimizer.state_dict(),
                'args': self.args
                }
        args = self.args
        file_name = "{}/{}/{}.ckpt".format(args.out_dir, args.model_name, args.run_id)
        try:
            torch.save(saved_params, file_name)
            logging.info("[INFO] Save model to [{}] Success!".format(file_name))
        except BaseException:
            logger.warning(("[WARN] Save model to [{}] Failed!".format(file_name))

    def load(self):
        pass

    def train_batch(self, squad_list):
        '''
        Args:
            squad_list -- a batch data
        Returns:
            loss -- loss
            prob_starts -- [b, slen]
            prob_ends -- [b, slen]
            qids -- [b] question ids
        '''
        inputs, span_starts, span_ends, qids = \
                self.model.batch_input_train(squad_list, self.args.device, Constant.padid)
        prob_starts, prob_ends = self.model(*inputs)
        loss = self.cal_loss(prob_starts, prob_ends, span_starts, span_ends)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 10)
        self.optimizer.step_and_update_lr()
        self.global_step += 1
        return loss.item(), prob_starts, prob_ends, qids

    def init_optimizer(self, state_dict=None):
        '''init a ScheduledOptimizer object
        Args:
            state_dict -- state_dict, may load from file
        Returns:
            optimizer
        '''
        optimizer = optim.Adam(self.params, betas=(0.8, 0.999), eps=1e-07)
        if state_dict is not None:
            optimizer.load_state_dict(state_dict)
        optimizer = ScheduledOptimizer(optimizer, self.args.encode_size, self.args.n_warmup_steps)
        self.optimizer = optimizer
        return optimizer

    @staticmethod
    def cal_loss(prob_starts, prob_ends, starts, ends):
        ''' calculate a batch's loss
        Args:
            prob_starts -- softmax probs, [b, plen]
            prob_ends -- softmax probs, [b, plen]
            starts -- [b]
            ends -- [b]
        Returns:
            loss -- avg loss, tensor
        '''
        prob_starts, prob_ends = torch.log(prob_starts), torch.log(prob_ends)
        loss_start = F.nll_loss(prob_starts, starts)
        loss_end = F.nll_loss(prob_ends, ends)
        loss = (loss_start + loss_end) / 2
        return loss

    @staticmethod
    def prob2answer(prob_starts, prob_ends, qids, qid2info):
        '''convert prob start-end to a real answer str
        Args:
            prob_starts -- span start prob, softmax
            prob_ends -- span end prob, softmax
            qids -- [b], question id
            qid2info -- {}
        Returns:
            answer_preds -- [], raw str
        '''
        pred_span, pred_score = get_span_from_probs(prob_starts, prob_ends)
        answer_preds = []
        for i, qid in enumerate(qids):
            info = qid2info[qid]
            span = pred_span[i][0]
            widx_start, widx_end = span[0], span[1]
            chidx_start = info['passage_token_offsets'][widx_start][0]
            # exclude the chidx_end char
            chidx_end = info['passage_token_offsets'][widx_end][1]
            answer_pred = info['passage'][chidx_start:chidx_end]
            answer_preds.append(answer_pred)
        return answer_preds

    @staticmethod
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
        answer_preds = MRCManager.prob2answer(prob_starts, prob_ends, qids, qid2info)
        f1 = 0
        em = 0
        total = 0
        for i, qid in enumerate(qids):
            info = qid2info[qid]
            answer_pred = answer_preds[i]
            f1 += f1_score(answer_pred, info['answer'])
            em += exact_match_score(answer_pred, info['answer'])
            total += 1
        f1 = 100.0 * f1 / total
        em = 100.0 * em / total
        return round(f1, 2), round(em, 2)


def prepare_embed_mat(word2idx, char2idx):
    word_embed_size = 300
    char_embed_size = 200
    word_mat = [np.random.normal(0, 1, word_embed_size) for i in range(len(word2idx))]
    char_mat = [np.random.normal(0, 1, char_embed_size) for i in range(len(char2idx))]
    word_mat = torch.FloatTensor(word_mat)
    char_mat = torch.FloatTensor(char_mat)
    return word_mat, char_mat


def go_train(manager, train_dataset, train_qid2info, opt):
    for e in range(500):
        print ("--------epoch={}----------".format(e+1))
        train_data_loader = get_data_loader(train_dataset, opt.batch_size, True)
        for batch in train_data_loader:
            loss, prob_starts, prob_ends, qids = manager.train_batch(batch)
            f1, em = manager.cal_performance(prob_starts, prob_ends, qids, train_qid2info)
            info = "{} loss={}, f1={}, em={}".format(
                        manager.global_step, round(loss, 2), f1, em)
            print (info)


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
    manager = MRCManager(opt, model)
    manager.init_optimizer()
    go_train(manager, train_dataset, train_qid2info, opt)


if __name__ == '__main__':
    print ("hello")
    test_main()
