import torch
import numpy as np
import pandas as pd

import os
import random

from models import *

if __name__ == "__main__":

    REST_THRESHOLD = 0.8 # if rest col > REST_THRESHOLD, then say the note is a rest.

    model = torch.load("./model_dir/temp_model_g.pt")


    while True:
        # initial = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # pitch
        #                     0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,  # octave
        #                     0]).reshape((1, 1, 24))  # rest
        # initial2 = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,  # pitch
        #                     0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,  # octave
        #                     0]).reshape((1, 1, 24))  # rest

        c_row = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                          0]).reshape((1, 24))
        d_row = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                          0]).reshape((1, 24))
        e_row = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                          0]).reshape((1, 24))
        f_row = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                          0]).reshape((1, 24))
        g_row = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                          0]).reshape((1, 24))
        cat_tensora = np.concatenate((c_row, d_row, e_row, f_row, g_row, g_row, f_row, e_row, d_row, c_row) * 10)
        cat_tensorb = np.concatenate((c_row, d_row, e_row, f_row, g_row, g_row, f_row, e_row, d_row, c_row) * 10)

        cat_tensora = cat_tensora[:-2]
        cat_tensorb = cat_tensorb[:-2]

        a = cat_tensora[np.newaxis, :,:]#np.concatenate((g_row, c_row) * 50, axis=1)
        b = cat_tensorb[np.newaxis, :,:]#np.concatenate((g_row, c_row) * 50, axis=1)

        # initial = np.random.random((24*100,)).reshape((1, 100, 24))
        print("", end="")
        for i in range(100):
            tensora = torch.Tensor(a)
            preda = model(tensora)

            tensorb = torch.Tensor(b)
            predb = model(tensorb)

            preda = np.array(preda.detach())[0, -1, :]
            predb = np.array(predb.detach())[0, -1, :]

            if preda[23] > REST_THRESHOLD:
                preda[23] = 1
                preda[0:23] = 0
            else:
                preda[0:12] = (preda[0:12] == np.max(preda[0:12]))
                preda[12:23] = (preda[12:23] == np.max(preda[12:23]))
                preda[23] = 0

            preda = np.reshape(preda, (1, 1, 24))
            a = np.concatenate((a, preda), axis=1)
            predb = np.reshape(predb, (1, 1, 24))
            b = np.concatenate((b, predb), axis=1)

        a = a[0]
        b = b[0]

        print()