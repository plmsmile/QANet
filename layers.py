#!/usr/bin/env python
#-*-coding: utf8-*-

'''
some common layers
@author: plm
@create: 2018-09-15 (Saturday)
@modified: 2018-09-15 (Saturday)
'''
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


def get_seq_pad_mask(seq, padid=0):
    '''get pad mask of a sequence
    Args:
        seq -- [b, slen]
        padid --
    Returns:
        pad_mask -- [b, slen]. pad is 1, nopad is 0.
    '''
    pad_mask = seq.eq(padid).float()
    return pad_mask


def get_matrix_mask(seq1_mask, seq2_mask):
    ''' get a matrix mask.
    passage-question(attention) or passage-passage mask(self-attention)
    Args:
        seq1_mask -- [b, len1], pad is 1, nopad is 0
        seq2_mask -- [b, len2], pad is 1, nopad is 0
    Returns:
        matrix_mask -- [b, len1, len2]. pad is 1, nopad is 0
    '''
    # pad is 0, nopad is 1
    seq1_keep = 1 - seq1_mask.unsqueeze(-1)
    seq2_keep = 1 - seq2_mask.unsqueeze(-1).transpose(1, 2)
    matrix_keep = torch.bmm(seq1_keep, seq2_keep)
    # pad is 1, nopad is 0
    matrix_mask = 1 - matrix_keep
    return matrix_mask


def mask_logits(target, mask, mask_value=-1e7):
    '''mask target tensor
    Args:
        target -- tensor
        mask -- tensor, same shape as target. mask place is 1, no mask is 0
    Returns:
        target_new --
    '''
    # return target.masked_fill_(mask, mask_value)
    return target * (1 - mask) + mask * mask_value


class DepthwiseSeparableConv(nn.Module):
    '''depthwise separable convolution'''

    def __init__(self, in_channels, out_channels, kernel_size, is_1d=True, bias=True):
        '''
        Args:
            in_channels -- input size
            out_channels -- output size
            kernel_size -- kernel size
            is_1d -- 1d conv or 2d conv
            bias -- bias
        '''
        super().__init__()
        conv = nn.Conv1d
        self.is_1d = is_1d
        if is_1d is False:
            conv = nn.Conv2d
        # depthwise conv. groups=in_channels
        self.depthwise_conv = conv(in_channels,
                                   in_channels,
                                   kernel_size,
                                   padding = kernel_size // 2,
                                   groups = in_channels)
        # pointwise conv. 1*1 conv. combine all channels' information
        self.pointwise_conv = conv(in_channels,
                                   out_channels,
                                   kernel_size = 1,
                                   padding = 0,
                                   bias = bias)

    def forward(self, inputs, show=False):
        ''' depthwise + pointwise
        Args:
            inputs -- [b, seqlen, input_size] or [b, seqlen, wordlen, input_size]
        Returns:
            res -- [b, seqlen, output_size] or [b, seqlen, wordlen, output_size]
        '''
        if self.is_1d:
            inputs = inputs.permute(0, 2, 1)
        else:
            inputs = inputs.permute(0, 3, 1, 2)
        depth_res = self.depthwise_conv(inputs)
        depth_res = F.relu(depth_res)
        point_res = self.pointwise_conv(depth_res)
        if self.is_1d:
            point_res = point_res.permute(0, 2, 1)
        else:
            point_res = point_res.permute(0, 2, 3, 1)
        return point_res


class HighwayNetwork(nn.Module):
    '''highway network'''

    def __init__(self, n_layer, input_size):
        '''
        Args:
            n_layer --
            input_size --
        '''
        super().__init__()
        self.n_layer = n_layer
        self.linears = nn.ModuleList([
                nn.Linear(input_size, input_size) for _ in range(self.n_layer)])
        self.gates = nn.ModuleList([
                nn.Linear(input_size, input_size) for _ in range(self.n_layer)])

    def forward(self, x):
        '''highway x
        Args:
            x -- a tensor, any shape
        Returns:
            x --
        '''
        for i in range(self.n_layer):
            gate = torch.sigmoid(self.gates[i](x))
            nonlinear = F.relu(self.linears[i](x))
            x = gate * nonlinear + (1 - gate) * x
        return x


class SublayerConnection(nn.Module):
    ''' LayerNorm(f(x) + x), f(LayerNorm(x)) + x '''

    def __init__(self, size, dropout=0.1, add_first=True):
        '''
        Args:
            size -- x input size
            dropout -- f dropout p
            add_first -- True: LN(f(x)+x), False: f(LN(x)) + x
        '''
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.add_first = add_first

    def forward(self, sublayer, params):
        ''' function + residual connection + layer norm
        Args:
            sublayer -- a sublayer whose output shape is the same as x
            params -- [] or tensor. sublayer's params. params[0] is the real data tensor
        Returns:
            res -- []
        '''
        # 1. resolve real data tenso
        x = None
        if isinstance(params, list):
            # multi params, the first one is the real data tensor
            x = params[0]
            attn_mask = params[1]
            has_param = True
        else:
            # only data tensor, no other params
            x = params
            params = [x]

        # 2. compute data
        res = x
        if self.add_first:
            residual_res = self.dropout(sublayer(*params)) + x
            res = self.norm(residual_res)
        else:
            norm_res = self.norm(x)
            params[0] = norm_res
            res = self.dropout(sublayer(*params)) + x
        return res


class PositionalEncoder(nn.Module):

    def __init__(self, embed_size, n_position):
        super().__init__()
        mat = []
        for pos in range(n_position):
            embed = []
            for i in range(embed_size):
                t = pos / np.power(10000, 2 * i / embed_size)
                embed.append(t)
            mat.append(embed)
        # dim 2i
        # mat = torch.Tensor(mat)
        mat = np.array(mat)
        mat[:, 0::2] = np.sin(mat[:, 0::2])
        # # dim 2i+1
        mat[:, 1::2] = np.cos(mat[:, 1::2])
        mat = torch.Tensor(mat)
        # mat = torch.FloatTensor(mat, requires_grad=False)
        self.pos_embedding = nn.Embedding.from_pretrained(mat, freeze=True)

    def forward(self, batch_size, seqlen):
        '''
        Args:
            batch_size -- b
            seqlen -- length
        Returns:
            pos_embeds -- [b, seqlen, embed_size]
        '''
        pos_idx = torch.arange(0, seqlen).to("cuda:1")
        pos_idx = pos_idx.repeat(batch_size, 1)
        pos_embeds = self.pos_embedding(pos_idx)
        return pos_embeds


class ScaledDotProductAttention(nn.Module):
    '''scaled dot-product attention'''

    def __init__(self, dkey, attn_dropout=0.1):
        '''
        Args:
            dkey -- key vector's size
            attn_dropout -- attention weights dropout
        '''
        super().__init__()
        self.scaled = np.power(dkey, 0.5)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, query, key, value, mask=None):
        ''' Q-K-V Attention
        Args:
            query -- [b, slen, dk]
            key -- [b, slen, dk]
            value -- [b, slen, dv]
            mask -- [b, slen, slen]. pad is 1, nopad is 0
        Returns:
            attention -- [b, slen, dv]
            attn_weights -- [b, slen, slen]
        '''
        # [b, slen, slen]
        score = torch.bmm(query, key.transpose(1, 2))
        score = score / self.scaled
        if mask is not None:
            score = mask_logits(score, mask)
        attn_weights = self.softmax(score)
        if mask is not None:
            attn_weights = mask_logits(attn_weights, mask, 0)
        attn_weights = self.dropout(attn_weights)
        attention = torch.bmm(attn_weights, value)
        return attention, attn_weights


class MultiHeadAttention(nn.Module):
    '''multi head attention'''

    def __init__(self, dmodel, dk=None, dv=None, nhead=1, dropout=0.1):
        '''
        Args:
            dmodel -- input and output size
            dk -- query-key size
            dv -- value size
            nhead --
            dropout -- dropout p
        '''
        super().__init__()
        if dk is None or dv is None:
            dk = dmodel
            dv = dmodel
        self.dmodel = dmodel
        self.dk = dk
        self.dv = dv
        self.nhead = nhead
        self.wq = nn.Linear(dmodel, dk * nhead)
        self.wk = nn.Linear(dmodel, dk * nhead)
        self.wv = nn.Linear(dmodel, dv * nhead)
        self.attention = ScaledDotProductAttention(dk)
        self.fc = nn.Linear(nhead * dv, dmodel)

    def forward(self, inputs, mask=None):
        '''
        Args:
            inputs -- [b, slen, dmodel]
            mask -- [b, slen, slen]. pad is 1, nopad is 0. self-attention mask
        Returns:
            outputs -- [b, slen, dmodel]
        '''
        dmodel, dk, dv, nhead = self.dmodel, self.dk, self.dv, self.nhead
        b, slen = inputs.size(0), inputs.size(1)

        q = self.wq(inputs).view(b, slen, nhead, dk)
        k = self.wk(inputs).view(b, slen, nhead, dk)
        v = self.wv(inputs).view(b, slen, nhead, dv)

        # [b*nhead, slen, dk]
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, slen, dk)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, slen, dk)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, slen, dv)
        # [b*n, slen, dv]
        if mask is not None:
            mask = mask.repeat(nhead, 1, 1)
        attention, attn_weights, = self.attention(q, k, v, mask)
        # [n, b, slen, dv]
        attention = attention.view(nhead, b, slen, dv)
        attention = attention.permute(1, 2, 0, 3).contiguous().view(b, slen, nhead*dv)
        # [b, slen, dmodel]
        outputs = self.fc(attention)
        return outputs


class PositioniseFeedForward(nn.Module):
    '''Position-wise Feed-Forward Network. max(0, xW1+b1)W2+b2'''
    def __init__(self, dmodel, dinner=None, dropout=0.1):
        '''
        Args:
            dmodel -- input and output size
            dinner -- inner layer size
            dropout --
        '''
        super().__init__()
        if dinner is None:
            dinner = dmodel
        self.linear_inner = nn.Linear(dmodel, dinner)
        self.linear_out = nn.Linear(dinner, dmodel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        Args:
            x -- [b, *, dmodel]
        Returns:
            output -- [b, *, dmodel]
        '''
        inner = F.relu(self.linear_inner(x))
        inner = self.dropout(inner)
        output = self.linear_out(inner)
        return output


class EncoderBlock(nn.Module):
    '''pos-embedding + convolution + self-attention + ffn'''

    def __init__(self, encode_size, n_conv=1, kernel_size=7, n_position=400):
        '''
        Args:
            encode_size --
            n_conv --
            kernel_size --
            n_position --
        '''
        super().__init__()
        self.pos_encoder = PositionalEncoder(encode_size, n_position)
        self.n_conv = n_conv
        self.kernel_size = kernel_size
        # n convolutions
        convs = []
        for i in range(n_conv):
            conv = DepthwiseSeparableConv(encode_size, encode_size, kernel_size)
            convs.append(conv)
        self.convs = nn.ModuleList(convs)
        # multi head attention
        self.attention = MultiHeadAttention(encode_size, nhead=8)
        # position-wise ffn
        self.ffn = PositioniseFeedForward(encode_size, encode_size*4)
        # f(layernorm(x)) + x
        self.connection = SublayerConnection(encode_size, add_first=False)

    def forward(self, inputs, inputs_mask=None):
        '''
        Args:
            inputs -- [b, slen, encode_size]
            inputs_mask -- [b, slen]. pad is 1, nopad is 0
        Returns:
            outputs -- [b, slen, encode_size]
        '''
        attn_mask = None
        inputs_keep = None
        if inputs_mask is not None:
            # [b, slen, slen], pad is 1, nopad is 0
            attn_mask = get_matrix_mask(inputs_mask, inputs_mask)
            # [b, slen, 1], pad is 0, nopad is 1
            inputs_keep = 1 - inputs_mask.unsqueeze(-1).float()
        b, seqlen = inputs.size(0), inputs.size(1)
        pos_embeds = self.pos_encoder(b, seqlen)
        inputs = inputs + pos_embeds
        # [b, slen, encode_size]
        for i, conv in enumerate(self.convs):
            inputs = self.connection(conv, inputs)
        # [b, slen, encode_size]
        attention_res  = self.connection(self.attention, [inputs, attn_mask])
        if inputs_keep is not None:
            # [b, slen, 1]
            attention_res = attention_res * inputs_keep
        # [b, slen, encode_size]
        outputs = self.connection(self.ffn, attention_res)
        if inputs_keep is not None:
            outputs = outputs * inputs_keep
        return outputs


class EncoderBlocks(nn.Module):

    def __init__(self, encode_size, n_block=1, n_conv=1, kernel_size=7, n_position=400):
        '''
        Args:
            encode_size -- the input and output size
            n_block -- n encoder blocks
            n_conv -- n convolutions of a block
            kernel_size -- convolution kernel size
            n_position --
        '''
        super().__init__()
        blocks = []
        for i in range(n_block):
            block = EncoderBlock(encode_size, n_conv, kernel_size, n_position)
            blocks.append(block)
        self.encoder_blocks = nn.ModuleList(blocks)

    def forward(self, inputs, inputs_mask=None):
        '''
        Args:
            inputs -- [b, slen, encode_size]
            inputs_mask -- [b, slen]. pad is 1, nopad is 0
        Returns:
            outputs -- [b, slen, encode_size]
        '''
        outputs = inputs
        for encoder in self.encoder_blocks:
            outputs = encoder(outputs, inputs_mask)
        return outputs


class CoAttention(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.score_linear = nn.Linear(3 * input_size, 1)

    def get_score(self, passage, question):
        '''compute similarity matrix. BiDAF's stratege: w[p, q, p*q]
        Args:
            passage -- [b, plen, d]
            question -- [b, qlen, d]
        Returns:
            score -- [b, plen, qlen]
        '''
        plen, qlen = passage.size(1), question.size(1)
        # [b, plen, qlen, d]
        passage = passage.unsqueeze(2).expand(-1, -1, qlen, -1)
        question = question.unsqueeze(1).expand(-1, plen, -1, -1)
        passage_question = torch.mul(passage, question)
        # concat the three infomation
        concat_info = torch.cat((passage, question, passage_question), -1)
        # compute score
        score = self.score_linear(concat_info).squeeze(-1)
        return score

    def forward(self, passage, question, passage_mask=None, question_mask=None):
        '''
        Args:
            passage -- [b, plen, d]
            question -- [b, qlen, d]
            passage_mask -- [b, plen]. pad is 1, nopad is 0
            question_mask -- [b, qlen]. pad is 1, nopad is 0
        Returns:
            p2q_attention -- [b, plen, d]
            coattention -- [b, plen, d]
        '''
        score = self.get_score(passage, question)
        need_mask = False
        mask = None
        if passage_mask is not None and question_mask is not None:
            need_mask = True
            # [b, plen, qlen], pad is 1, nopad is 0
            mask = get_matrix_mask(passage_mask, question_mask)
            score = mask_logits(score, mask)

        # [b, plen, qlen]
        p2q_weights = F.softmax(score, dim=2)
        # [b, qlen, plen]
        q2p_weights = F.softmax(score.transpose(1, 2), dim=2)
        if need_mask:
            p2q_weights = mask_logits(p2q_weights, mask, 0)
            q2p_weights = mask_logits(q2p_weights, mask.transpose(1, 2), 0)

        # [b, plen, d]
        p2q_attention = torch.bmm(p2q_weights, question)
        # [b, qlen, d]
        q2p_attention = torch.bmm(q2p_weights, passage)
        # [b, plen, d] dcn coattention
        coattention = torch.bmm(p2q_weights, q2p_attention)
        return p2q_attention, coattention


class InputEmbedding(nn.Module):
    '''combine word embedding and char embedding'''

    def __init__(self,
                 word_embed_size,
                 char_embed_size,
                 dropout_word=0.1,
                 dropout_char=0.05,
                 kernel_size=5):
        '''
        Args:
            word_embed_size -- word vector dim
            char_embed_size -- char vector dim
            dropout_word -- dropout prob for char embedding
            dropout_char -- dropout prob for char embedding
            kernel_size -- convolution kernel size for char embedding
        '''
        super().__init__()
        self.dropout_word = dropout_word
        self.dropout_char = dropout_char
        self.conv2d = DepthwiseSeparableConv(char_embed_size,
                                             char_embed_size,
                                             kernel_size,
                                             is_1d = False)
        self.highway = HighwayNetwork(2, word_embed_size + char_embed_size)

    def forward(self, word_embeds, char_embeds, show=False):
        '''
        Args:
            word_embeds -- [b, seqlen, word_embed_size]
            char_embeds -- [b, seqlen, wordlen, char_embed_size]
        Returns:
            embeds -- [b, seqlen, word_embed_size + char_embed_size]
        '''
        # 1. char embeds
        char_embeds = F.dropout(char_embeds, p=self.dropout_char, training=self.training)
        # [b, slen, wlen, dchar]
        char_embeds = self.conv2d(char_embeds, show)
        char_embeds = F.relu(char_embeds)
        # [b, slen, dchar], choose the max char of every word
        char_embeds, idxs = torch.max(char_embeds, dim=2)

        # 2. word embeds
        word_embeds = F.dropout(word_embeds, p=self.dropout_word, training=self.training)

        # 3. concat word-char, highway network
        embeds = torch.cat([word_embeds, char_embeds], dim=2)
        embeds = self.highway(embeds)
        return embeds
