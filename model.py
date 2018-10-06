#!/usr/bin/env python
#-*-coding: utf8-*-

'''
QANet model
@author: plm
@create: 2018-09-08 (Saturday)
@modified: 2018-09-08 (Saturday)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import InputEmbedding, DepthwiseSeparableConv, EncoderBlocks, CoAttention
from layers import get_seq_pad_mask, mask_logits
from data_helper import pad_batch_squad


class QANet(nn.Module):
    '''QANet model'''
    def __init__(self,
                 word_mat, char_mat, dropout, dropout_char, max_seqlen,
                 encode_size):
        super().__init__()
        freeze = True
        word_freeze = freeze
        char_freeze = False
        self.word_mat = nn.Embedding.from_pretrained(word_mat, freeze=word_freeze)
        self.char_mat = nn.Embedding.from_pretrained(char_mat, freeze=char_freeze)
        self.word_mat.padding_idx = 0
        self.char_mat.padding_idx = 0
        dword = self.word_mat.embedding_dim
        dchar = self.char_mat.embedding_dim
        # input embedding
        self.input_embedding = InputEmbedding(dword, dchar, dropout, dropout_char)
        # convert embed_size to encode_size
        self.passage_conv = DepthwiseSeparableConv(dword+dchar, encode_size, 5)
        self.question_conv = DepthwiseSeparableConv(dword+dchar, encode_size, 5)
        # embedding encoder
        self.embedding_encoder = EncoderBlocks(encode_size)
        # attention layer
        self.attention = CoAttention(encode_size)
        self.concat_info_conv = DepthwiseSeparableConv(4*encode_size, encode_size, 5)
        # model encoder
        self.model_encoder = EncoderBlocks(encode_size, 2, 2)
        # output
        self.compute_score_start = nn.Linear(2*encode_size, 1)
        self.compute_score_end = nn.Linear(2*encode_size, 1)

    def forward(self, passages, questions, passage_chars=None, question_chars=None):
        '''
        Args:
            passages -- [b, slen_p]
            questions -- [b, slen_q]
            passage_chars -- [b, slen_p, wlen_p]
            question_chars -- [b, slen_q, wlen_q]
        Returns:
            prob_start -- [b, slen_p], softmax prob
            prob_end -- [b, slen_p], softmax prob
        '''
        # 0 mask. pad is 1, nopad is 0
        passage_mask = get_seq_pad_mask(passages)
        question_mask = get_seq_pad_mask(questions)
        # 1. input embedding
        questions = self.word_mat(questions)
        question_chars = self.char_mat(question_chars)
        passages = self.word_mat(passages)
        passage_chars = self.char_mat(passage_chars)
        # [b, qlen, dword+dchar]
        questions = self.input_embedding(questions, question_chars)
        # [b, plen, dword+dchar]
        passages = self.input_embedding(passages, passage_chars, show=False)
        # [b, plen, encode_size]
        passages = self.passage_conv(passages)
        questions = self.question_conv(questions)

        # 2. embedding encoder, [b, qlen/plen, encode_size]
        questions = self.embedding_encoder(questions, question_mask)
        passages = self.embedding_encoder(passages, passage_mask)

        # 3. attention layer
        attention, coattention = \
                self.attention(passages, questions, passage_mask, question_mask)
        # may add coattention, the paper doesn't
        concat_info = [passages, attention, passages * attention, passages * coattention]
        concat_info = torch.cat(concat_info, dim=2)
        passage_with_question = self.concat_info_conv(concat_info)

        # 4. model encoder, [b, qlen/plen, encode_size]
        passage_info = self.model_encoder(passage_with_question, passage_mask)
        passage_info_start = self.model_encoder(passage_info, passage_mask)
        passage_info_end = self.model_encoder(passage_info_start, passage_mask)

        # 5. output layer, [b, plen, 2*encode_size]
        info_start = torch.cat([passage_info, passage_info_start], dim=2)
        info_end = torch.cat([passage_info, passage_info_end], dim=2)
        score_start = self.compute_score_start(info_start).squeeze(-1)
        score_end = self.compute_score_start(info_end).squeeze(-1)
        prob_start = F.softmax(score_start, dim=1)
        prob_end = F.softmax(score_end, dim=1)
        return prob_start, prob_end

    @staticmethod
    def batch_input_train(squad_list, device=torch.device("cpu"), padid=0):
        '''make model inputs data
        Args:
            squad_list --
            device -- cpu or gpu data
        Returns:
            inputs -- p, q, pch, qch
            span_starts -- [b]
            span_ends -- [b]
            qids -- [b]
        '''
        res = pad_batch_squad(squad_list, padid)
        tensors, qids = res[:-1], res[-1]
        inputs = [tensor.to(device) for tensor in tensors]
        passages, questions, passage_chars, question_chars, span_starts, span_ends = inputs
        inputs = [passages, questions, passage_chars, question_chars]
        return inputs, span_starts, span_ends, qids


