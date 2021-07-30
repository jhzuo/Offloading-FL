#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


# def FedAvg(w):
#      w_avg = copy.deepcopy(w[0])
#      for k in w_avg.keys():
#          for i in range(1, len(w)):
#              w_avg[k] += w[i][k]
#          w_avg[k] = torch.div(w_avg[k], len(w))
#      return w_avg






def FedAvg(states, L): 

    gloabl_state = dict()


    for key in states[0]:
        for i in range(len(L)):
            if not key in gloabl_state.keys():
                count_D = L[i]
                gloabl_state[key] = L[i] * states[i][key]
            else:
                count_D += L[i]
                gloabl_state[key] += L[i] * states[i][key]

        gloabl_state[key] /= count_D

    return gloabl_state
