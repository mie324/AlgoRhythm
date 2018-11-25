import torch
import numpy as np
import pandas as pd

import os
import random

from models import *
from main import add_shifted_copies
from midi_converter import convert_array_to_midi2

if __name__ == "__main__":

    REST_THRESHOLD = 0.8 # if rest col > REST_THRESHOLD, then say the note is a rest.

    #model = torch.load("./model_dir/epoch3000_algorhythm_Namespace(batch_size=4, dim_hidden=100, emb_dim=100, epochs=4000, eval_every=100, loss_fn='mse', lr=0.001, memory=7, model='ffnn', num_hidden_layers=3, optimizer='adam', rnn_hidden_dim=100).pt")
    model = torch.load("./model_dir/model.pt")
    variable = True

    while variable:
        # initial = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # pitch
        #                     0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,  # octave
        #                     0]).reshape((1, 1, 24))  # rest
        # initial2 = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,  # pitch
        #                     0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,  # octave
        #                     0]).reshape((1, 1, 24))  # rest

        c_row = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                          0,1]).reshape((1, 25))
        cs_row = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                          0,1]).reshape((1, 25))
        d_row = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                          0,1]).reshape((1, 25))
        ds_row = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                          0,1]).reshape((1, 25))
        e_row = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                          0,1]).reshape((1, 25))
        f_row = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                          0,1]).reshape((1, 25))
        fs_row = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                          0,1]).reshape((1, 25))
        g_row = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                          0,1]).reshape((1, 25))
        gs_row = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                          0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                          0,1]).reshape((1, 25))
        a_row = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                          0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                          0,1]).reshape((1, 25))
        as_row = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                          0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                          0,1]).reshape((1, 25))
        b_row = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                          0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                          0,1]).reshape((1, 25))
        cc_row = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                          0,1]).reshape((1, 25))

        a_single = np.concatenate((c_row, d_row, e_row, f_row, g_row, g_row, f_row, e_row, d_row, c_row) * 10)
        b_single = np.concatenate((c_row, d_row, e_row, f_row, g_row, g_row, f_row, e_row, d_row, c_row) * 10)
        # a_single = np.concatenate((d_row, e_row, fs_row, g_row, a_row, a_row, g_row, fs_row, e_row, d_row) * 10)
        # b_single = np.concatenate((d_row, e_row, fs_row, g_row, a_row, a_row, g_row, fs_row, e_row, d_row) * 10)

        # a_single = np.random.random((20, 24))
        # b_single = a_single

        # a = cat_tensora[np.newaxis, :,:]#np.concatenate((g_row, c_row) * 50, axis=1)
        # b = cat_tensorb[np.newaxis, :,:]#np.concatenate((g_row, c_row) * 50, axis=1)

        # a_with_shift = cat_tensora_with_shift[np.newaxis, :,:]#np.concatenate((g_row, c_row) * 50, axis=1)
        # b_with_shift = cat_tensorb_with_shift[np.newaxis, :,:]#np.concatenate((g_row, c_row) * 50, axis=1)

        print("", end="")
        for i in range(500):
            a = add_shifted_copies(a_single, 7)[np.newaxis, :, :]
            b = add_shifted_copies(b_single, 7)[np.newaxis, :, :]

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

            preda = np.reshape(preda, (1, 25))
            a_single = np.concatenate((a_single, preda), axis=0)
            predb = np.reshape(predb, (1, 25))
            b_single = np.concatenate((b_single, predb), axis=0)

        a_single
        b_single

        # Prints the music tensors as midi
        convert_array_to_midi2(a_single[100:], "./generated_midi/generated_a.mid")
        convert_array_to_midi2(b_single[100:], "./generated_midi/generated_b.mid")

        print("Complete")
        variable = False
