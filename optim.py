#!/usr/bin/env python
#-*-coding: utf8-*-

'''
a wrapper class for optimizer
@author: plm
@create: 2018-09-25
@modified: 2018-09-25
'''

import numpy as np

class ScheduledOptimizer(object):
    '''a simple wrapper of pytorch optimizer, transformer optimizer style'''

    def __init__(self, optimizer, dmodel=128, n_warmup_steps=4000):
        '''
        Args:
            optimizer -- a pytorch optimizer object
            dmodel -- encoder-block output-size, transformer-512, qanet-128
            n_warmup_steps --
        '''
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.current_step = 0
        self.init_lr = np.power(dmodel, -0.5)

    def _get_lr_scale(self):
        '''lr update scale, lr = lr * scale'''
        a = np.power(self.current_step, -0.5)
        b = self.current_step * np.power(self.n_warmup_steps, -1.5)
        return min(a, b)

    def _update_learning_rate(self):
        '''update lr per step'''
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def zero_grad(self):
        '''zero out the gradients by the inner optimizer'''
        self._optimizer.zero_grad()

    def step_and_update_lr(self):
        '''step with the inner optimizer'''
        self._update_learning_rate()
        self._optimizer.step()

    def state_dict(self):
        d = {'state_dict':self._optimizer.state_dict()}
        params = {"n_warmup_steps":self.n_warmup_steps,
                  "init_lr": self.init_lr,
                  "current_step": self.current_step,}
        d["params"] = params
        return d

    @staticmethod
    def get_instance(optimizer, params):
        instance = ScheduledOptimizer(optimizer)
        instance.n_warmup_steps = params['n_warmup_steps']
        instance.current_step = params['current_step']
        instance.init_lr = params['init_lr']
        return instance


