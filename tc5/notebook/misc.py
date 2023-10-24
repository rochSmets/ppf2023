# -*- coding: utf-8 -*-

# sc functions needed to settle a numerical scheme
# ============================================


import numpy as np


def addGhosts(x, num_of_ghosts=1):
    x_wide = np.ones(x.shape[0]+2*num_of_ghosts)*(-1)
    x_wide[num_of_ghosts:-num_of_ghosts] = x
    return x_wide


def fillGhosts(x, num_of_ghosts=1, periodic=True):
    if periodic:
        x[:num_of_ghosts] = x[-2*num_of_ghosts:-num_of_ghosts]
        x[-num_of_ghosts:] = x[num_of_ghosts:2*num_of_ghosts]
    return x
