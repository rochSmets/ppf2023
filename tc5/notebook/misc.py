# -*- coding: utf-8 -*-



# sc functions needed to settle a numerical scheme
# ============================================



import numpy as np


def addGhosts(x):
    x_wide = np.zeros(x.shape[0]+2)
    x_wide[1:-1] = x
    return x_wide

def fillGhosts(x, periodic = True):
    if periodic :
        x[0] = x[-2]
        x[-1] = x[1]
    return x

