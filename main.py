import torch
import numpy as np
import pandas as pd
import torch.optim as optim

import torchtext
from torchtext import data

import argparse
import os
import random

from midi_converter import files_to_dataloader, batch_to_tensor, files_to_cat_tensor_dataloader, super_ez_trn_example_dataloader

from models import RNN, FFNN

seed=1
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


TRN_PATHS = ["./midi/bach_wtc1/Fugue{}.mid".format(i) for i in [1,2,3,4,5,7,8,9,10,11,13,14,15,16,17,18,20,21,22,23,24]]#range(1, 12+1)]
VAL_PATHS = []




def main(args):




    #trn_bucketer = files_to_bucketiterator(TRN_PATHS, args.batch_size)
    #trn_loader = files_to_dataloader(TRN_PATHS, args.batch_size, first_voice_only=True, has_rest_col=True, shuffle=True)

    # trn_loader = files_to_cat_tensor_dataloader(TRN_PATHS, first_voice_only=False, has_rest_col=True, shuffle=True)
    trn_loader = super_ez_trn_example_dataloader(TRN_PATHS, first_voice_only=False, has_rest_col=True, shuffle=True)


    print("Hyperparameters:\n{}".format(args))

    model, loss_fnc, optimizer = load_model(args)
    n_trn_examples = 12
    n_steps_per_epoch = int(np.ceil(n_trn_examples / args.batch_size))
    n_entries = int(np.ceil(args.epochs * n_steps_per_epoch / args.eval_every))
    trn_acc_arr = np.zeros(n_entries)
    val_acc_arr = np.zeros(n_entries)
    best_val_acc = -1

    t = 0  # used to count batch number putting through the model
    for epoch in range(args.epochs):
        accum_loss = 0
        tot_corr = 0
        denominator = 0
        for i, batch in enumerate(trn_loader):
            tensor = batch['data']
            optimizer.zero_grad()

            predictions = model(tensor)

            if args.model == "rnn":
                predictions = predictions[:, :-1, :] #remove last prediction, because we don't know the "next note"

            if args.model == "ffnn":
                label = tensor[:, args.memory:, :] # remove first prediction, because the fnn doesn't predict the first `memory` notes
            elif args.model == "rnn":
                label = tensor[:, 1:, :] # remove first prediction, because the rnn doesn't predict the first note

            batch_loss = loss_fnc(input=predictions, target=label.float())
            accum_loss += batch_loss
            batch_loss.backward()
            optimizer.step()

            corr = (binarize_pred(np.array(predictions.detach())[0]) == label)[0]
            corr = np.all(np.array(corr), axis=1)
            tot_corr += int(corr.sum())
            denominator += corr.shape[0]
            print("epoch {:>4d}, loss {:>.6f}, acc {:>.6f}".format(epoch, float(batch_loss), corr.sum()/len(corr)))
            if False and t % args.eval_every == 0: #/make validation work
                val_acc, val_loss = evaluate(model, val_bucketer, loss_fnc)
                trn_acc_arr[t // args.eval_every] = float(mov_avg_corr) / denominator
                val_acc_arr[t // args.eval_every] = val_acc

                print(
                    "Epoch: {:>3d}, Step: {:>7d} | Train acc: {:.6f} | Loss: {:.6f} | Val acc: {:.6f}"
                        .format(epoch + 1,
                                  t + 1,
                                  trn_acc_arr[t // args.eval_every],
                                  accum_loss / args.eval_every,
                                  val_acc))
                if val_acc > best_val_acc:
                    torch.save(model, "./model_dir/model_{}.pt".format(args.model))
                    best_val_acc = val_acc

                accum_loss = 0
                mov_avg_corr = 0
                denominator = 0
            t = t + 1
    #print("Train acc: {}".format(float(tot_corr) / len(trn_loader.dataset)))
    torch.save(model, "./model_dir/temp_model.pt")

    #/make validation work
    # print("Model with best val acc:")
    # model_best = torch.load("./model_dir/model_{}.pt".format(args.model))
    # trn_acc_best, trn_loss_best = evaluate(model_best, trn_loader, loss_fnc)
    # val_acc_best, val_loss_best = evaluate(model_best, val_bucketer, loss_fnc)
    # tst_acc_best, tst_loss_best = evaluate(model_best, tst_bucketer, loss_fnc)
    # print("\t\tAcc\t\tLoss")
    # print("trn\t\t{:>.3f}\t{:>.6f}".format(trn_acc_best, trn_loss_best))
    # print("val\t\t{:>.3f}\t{:>.6f}".format(val_acc_best, val_loss_best))
    # print("tst\t\t{:>.3f}\t{:>.6f}".format(tst_acc_best, tst_loss_best))


######


# def binarize_pred_old(pred): #/hardcoded rn
#     REST_THRESHOLD = 0.8
#     for i in range(pred.shape[0]):
#         if pred[i,23] > REST_THRESHOLD:
#             pred[i,23] = 1
#             pred[i,0:23] = 0
#         else:
#             pred[i,0:12] = (pred[i,0:12] == np.max(pred[i,0:12]))
#             pred[i,12:23] = (pred[i,12:23] == np.max(pred[i,12:23]))
#             pred[i,23] = 0
#     return pred

def binarize_pred(pred): #/hardcoded rn
    REST_THRESHOLD = 0.8
    is_note = pred[:, 23:24] < REST_THRESHOLD
    pitch = (pred[:,0:12] == np.max(pred[:,0:12], axis=1).reshape(-1,1))
    octave = (pred[:,12:23] == np.max(pred[:,12:23], axis=1).reshape(-1,1))
    pred[:, 0:12] = is_note * pitch
    pred[:, 12:23] = is_note * octave
    pred[:,23:24] =  1 - is_note
    return pred

def load_model(args):
    if args.model == 'rnn':
        model = RNN(24, 200, 24)
    elif args.model == 'ffnn':
        model = FFNN(24, 100, 24, memory=args.memory)
    else:
        raise Exception("Only rnn and ffnn model type currently supported")

    if args.loss_fn == 'mse':
        loss_fn = torch.nn.MSELoss()
    else:
        raise Exception("Only mse loss function currently supported")

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise Exception("Only adam optimizer currently supported")
    return model, loss_fn, optimizer

def evaluate(model, bucketer, loss_fnc):
    total_corr = 0
    total_loss = 0
    for i, batch in enumerate(bucketer):
        feats, lengths, label = batch.Text[0], batch.Text[1], batch.Label
        predictions = model(feats, lengths)
        corr = ((predictions > 0.5).reshape(-1) == label.byte())
        total_corr += int(corr.sum())

        loss = loss_fnc(input=predictions.squeeze(), target=label.float())
        total_loss += loss

    acc = float(total_corr) / len(bucketer.dataset)
    avg_loss = total_loss / len(bucketer.dataset)
    return acc, avg_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--model', type=str, default="ffnn")
    parser.add_argument('--emb_dim', type=int, default=100)
    parser.add_argument('--rnn_hidden_dim', type=int, default=100)
    parser.add_argument('--loss_fn', type=str, default="mse")
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--memory', type=int, default=3)

    parser.add_argument('--eval_every', type=int, default=100)

    args = parser.parse_args()

    main(args)