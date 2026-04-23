import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class ProgressBar():
    def __init__(self, iteration, bin = 100):
        self._iteration = iteration
        self.postfix = ""
        self.bin = bin

    def __iter__(self):
        self.len_pre_print = 0
        for i, item in enumerate(self._iteration):
            self.i = i
            self.show()
            yield item
        print()
    
    def show(self):
        percentage = self.i/len(self._iteration)
        progress = "\r["+"="*round(percentage*self.bin) + "-"*(self.bin-round(percentage*self.bin))+f"] {round(percentage*100)}% "+ self.postfix 
        len_current_print = len(progress)
        end_blank = (self.len_pre_print-len_current_print) if len_current_print<self.len_pre_print else 0
        print(progress + " "*end_blank, end="")
        self.len_pre_print =  len_current_print
        
    def set_postfix(self, postfix={}):
        postfix_ls = []
        for key, value in postfix.items():
            postfix_ls.append(f" {key} : {value}")
        if len(postfix_ls)>0:
            self.postfix = "[" + ",".join(postfix_ls) + "]"
            self.show()
