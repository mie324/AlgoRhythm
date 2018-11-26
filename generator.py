import torch
import numpy as np
import pandas as pd

import os
import random

from models import *
from main import add_shifted_copies
from midi_converter import convert_array_to_midi_CNN

if __name__ == "__main__":

    REST_THRESHOLD = 0.8 # if rest col > REST_THRESHOLD, then say the note is a rest.

    #model = torch.load("./model_dir/epoch3000_algorhythm_Namespace(batch_size=4, dim_hidden=100, emb_dim=100, epochs=4000, eval_every=100, loss_fn='mse', lr=0.001, memory=7, model='ffnn', num_hidden_layers=3, optimizer='adam', rnn_hidden_dim=100).pt")
    model = torch.load("./model_dir/model_CNN.pt")
    variable = True

    while variable:
        # initial = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # pitch
        #                     0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,  # octave
        #                     0]).reshape((1, 1, 24))  # rest
        # initial2 = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,  # pitch
        #                     0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,  # octave
        #                     0]).reshape((1, 1, 24))  # rest

        # c_row = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #                   0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        #                   0,1]).reshape((1, 25))
        # cs_row = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #                   0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        #                   0,1]).reshape((1, 25))
        # d_row = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #                   0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        #                   0,1]).reshape((1, 25))
        # ds_row = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        #                   0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        #                   0,1]).reshape((1, 25))
        # e_row = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        #                   0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        #                   0,1]).reshape((1, 25))
        # f_row = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        #                   0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        #                   0,1]).reshape((1, 25))
        # fs_row = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        #                   0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        #                   0,1]).reshape((1, 25))
        # g_row = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        #                   0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        #                   0,1]).reshape((1, 25))
        # gs_row = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        #                   0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        #                   0,1]).reshape((1, 25))
        # a_row = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        #                   0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        #                   0,1]).reshape((1, 25))
        # as_row = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        #                   0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        #                   0,1]).reshape((1, 25))
        # b_row = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        #                   0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        #                   0,1]).reshape((1, 25))
        # cc_row = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #                   0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        #                   0,1]).reshape((1, 25))

        c_row = np.zeros((1, 134))
        cs_row = np.zeros((1, 134))
        d_row = np.zeros((1, 134))
        ds_row = np.zeros((1, 134))
        e_row = np.zeros((1, 134))
        f_row = np.zeros((1, 134))
        fs_row = np.zeros((1, 134))
        g_row = np.zeros((1, 134))
        gs_row = np.zeros((1, 134))
        a_row = np.zeros((1, 134))
        as_row = np.zeros((1, 134))
        b_row = np.zeros((1, 134))
        cc_row = np.zeros((1, 134))

        c_row[0, 0+5*11] = 1
        cs_row[0, 1+5*11] = 1
        d_row[0, 2+5*11] = 1
        ds_row[0, 3+5*11] = 1
        e_row[0, 4+5*11] = 1
        f_row[0, 5+5*11] = 1
        fs_row[0, 6+5*11] = 1
        g_row[0, 7+5*11] = 1
        gs_row[0, 8+5*11] = 1
        a_row[0, 9+5*11] = 1
        as_row[0, 10+5*11] = 1
        b_row[0, 11+5*11] = 1
        cc_row[0, 1+6*11] = 1

        c_row[0, 133] = 1
        cs_row[0, 133] = 1
        d_row[0, 133] = 1
        ds_row[0, 133] = 1
        e_row[0, 133] = 1
        f_row[0, 133] = 1
        fs_row[0, 133] = 1
        g_row[0, 133] = 1
        gs_row[0, 133] = 1
        a_row[0, 133] = 1
        as_row[0, 133] = 1
        b_row[0, 133] = 1
        cc_row[0, 133] = 1

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
            a = add_shifted_copies(a_single, 8)[np.newaxis, :, :]
            b = add_shifted_copies(b_single, 8)[np.newaxis, :, :]

            tensora = torch.Tensor(a)
            preda = model(tensora)

            tensorb = torch.Tensor(b)
            predb = model(tensorb)

            preda = preda.unsqueeze(0)
            predb = predb.unsqueeze(0)

            preda = np.array(preda.detach())[0, -1, :]
            predb = np.array(predb.detach())[0, -1, :]

            if preda[132] > REST_THRESHOLD:
                preda[132] = 1
                preda[0:132] = 0
            else:
                preda[0:132] = (preda[0:132] == np.max(preda[0:132]))
                preda[132] = 0

            preda = np.reshape(preda, (1, 134))
            a_single = np.concatenate((a_single, preda), axis=0)
            predb = np.reshape(predb, (1, 134))
            b_single = np.concatenate((b_single, predb), axis=0)

        a_single
        b_single

        # Prints the music tensors as midi
        convert_array_to_midi_CNN(a_single[100:], "./generated_midi/generated_a.mid")
        convert_array_to_midi_CNN(b_single[100:], "./generated_midi/generated_b.mid")

        print("Complete")
        variable = False
