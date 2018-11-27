import torch
import numpy as np
import pandas as pd
import torch.optim as optim

import torchtext
from torchtext import data

import argparse
import os
import random
from midi_converter import file_to_tensor, MIDI_PITCHES, MIDI_OCTAVES
import pickle

from dataset import MusicDataset

from torch.utils.data import DataLoader


from models import RNN, FFNN, CNN3D

seed=1
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


PATHS = []
PATHS.extend(["./midi/bach_wtc1/Prelude{}.mid".format(i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 21, 22, 23, 24]])
PATHS.extend(["./midi/bach_wtc1/Fugue{}.mid".format(i) for i in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24]])
PATHS.extend(["./midi/bach_wtc2/Prelude{}.mid".format(i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
PATHS.extend(["./midi/bach_wtc2/Fugue{}.mid".format(i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
PATHS.extend(["./midi/bach_2partinventions/invent{}.mid".format(i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
PATHS.extend(["./midi/bach_3partinventions/sinfon{}.mid".format(i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
# PATHS.extend(["./midi/bach_goldberg_variations_complete.mid"])




VERBOSE = True

#/todo try: concat columns with difference in pitch

# splits list of path files into trn, val, tst lists, according to the sizes.
def trn_val_tst_split(pathlist, trn_size, val_size, tst_size):
    total = float(trn_size + val_size + tst_size)
    val_size = round(val_size / total * len(pathlist))
    tst_size = round(tst_size / total * len(pathlist))
    trn_size = len(pathlist) - val_size - tst_size

    pathlist = np.random.permutation(pathlist)
    trn_pathlist = pathlist[:trn_size]
    val_pathlist = pathlist[trn_size : trn_size+val_size]
    tst_pathlist = pathlist[trn_size+val_size :]

    return trn_pathlist, val_pathlist, tst_pathlist

def add_shifted_copies(tensor, num_copies):
    tensor_2 = tensor
    for i in range(num_copies - 1):
        tensor_2 = tensor_2[:-1]
        tensor_2 = np.concatenate((tensor_2, tensor[(i + 1):]), axis=1)
    return tensor_2

# given a list of midi file paths, produces a DataLoader with those midi files
def files_to_dataloader(pathlist, concat=True, num_copies=1, first_voice_only=False, three_d_tensor=True):
    if not concat:
        raise Exception("cat_tensor=False not supported")

    if not three_d_tensor:
        tensor_shape = (0, len(MIDI_PITCHES) + len(MIDI_OCTAVES) + 2) #2 extra cols for the length and rest
        cat_tensor = np.zeros(tensor_shape)
        for path in pathlist:
            if VERBOSE:
                print("Processing {} ...".format(path))
            t = file_to_tensor(path, first_voice_only=first_voice_only, three_d_tensor=three_d_tensor)
            cat_tensor = np.concatenate((cat_tensor, t), axis=0)

        cat_tensor_2 = add_shifted_copies(cat_tensor, num_copies)
        if args.divide_data_pieces != 1:
            raise Exception("not supported yet")
        dataset = MusicDataset([cat_tensor_2], [cat_tensor])
        data_loader = DataLoader(dataset, batch_size=1)

    else: #three_d_tensor == True
        if num_copies != 1:
            print("Warning, ignoring num_copies due to three_d_tensor=True")

        tensor_shape = (0, len(MIDI_PITCHES), len(MIDI_OCTAVES))
        cat_tensor = np.zeros(tensor_shape)
        cat_rest_tensor = np.zeros((0,))
        cat_length_tensor = np.zeros((0,))

        for path in pathlist:
            if VERBOSE:
                print("Processing {} ...".format(path))
            notes_tensor, rests_tensor, lengths_tensor = file_to_tensor(path, first_voice_only=first_voice_only, three_d_tensor=three_d_tensor)

            cat_tensor = np.concatenate((cat_tensor, notes_tensor), axis=0)
            cat_rest_tensor = np.concatenate((cat_rest_tensor, rests_tensor))
            cat_length_tensor = np.concatenate((cat_length_tensor, lengths_tensor))

        dataset_list = []
        divided_length = round(float(np.floor(len(cat_tensor) / args.divide_data_pieces)))
        for i in range(args.divide_data_pieces):
            temp_cat_tensor = cat_tensor[i * divided_length : (i+1) * divided_length]
            temp_cat_rest_tensor = cat_rest_tensor[i * divided_length : (i+1) * divided_length]
            temp_cat_length_tensor = cat_length_tensor[i * divided_length : (i+1) * divided_length]
            dataset_list.append((temp_cat_tensor, temp_cat_rest_tensor, temp_cat_length_tensor))
        dataset = MusicDataset(dataset_list)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return data_loader

def super_ez_trn_example_dataloader():

    c_row = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                      0,1]).reshape((1, 25))
    d_row = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                      0,1]).reshape((1, 25))
    e_row = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                      0,1]).reshape((1, 25))
    f_row = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                      0,1]).reshape((1, 25))
    g_row = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                      0,1]).reshape((1, 25))
    cat_tensor = np.concatenate((c_row, d_row, e_row, f_row, g_row, g_row, f_row, e_row, d_row, c_row)*50)
    dataset = MusicDataset([cat_tensor])
    data_loader = DataLoader(dataset, batch_size=1)
    return data_loader

def splits(paths_tuple, args):
    loader_tuple = ()
    for paths in paths_tuple:
        loader_tuple += (files_to_dataloader(paths, concat=args.concat, num_copies=args.memory,
                                            first_voice_only=args.first_voice_only, three_d_tensor=(args.model == 'cnn3d')),)
    return loader_tuple


def main(args):
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if args.concat:
        has_saved_dataloaders = os.path.isfile("./output/loaders.pkl")
        if args.overwrite_cached_loaders or not has_saved_dataloaders:
            trn_paths, val_paths, tst_paths = trn_val_tst_split(PATHS, 0.85, 0.15, 0)
            trn_loader, val_loader, tst_loader = splits((trn_paths, val_paths, tst_paths), args)

            with open("./output/loaders.pkl", 'wb') as f:
                pickle.dump([trn_loader, val_loader, tst_loader], f, protocol=-1)
        else:
            with open("./output/loaders.pkl", 'rb') as f:
                trn_loader, val_loader, tst_loader = pickle.load(f)

    else:
        raise Exception("Not concatenating is not yet supported.")

    print("Hyperparameters:\n{}".format(args))

    model, loss_fnc, optimizer = load_model(args)
    n_trn_examples = 12
    n_steps_per_epoch = int(np.ceil(n_trn_examples / args.batch_size))
    n_entries = int(np.ceil(args.epochs * n_steps_per_epoch / args.eval_every))
    trn_acc_arr = np.zeros(n_entries)
    best_val_acc = -1

    t = 0  # used to count batch number putting through the model
    for epoch in range(args.epochs):
        accum_loss = 0
        tot_corr = 0
        denominator = 0
        for i, batch in enumerate(trn_loader):
            tensor = batch['data']
            actual = batch['label']
            optimizer.zero_grad()

            predictions = model(tensor)
            num_notes, batch_loss, acc = get_evaluation_stats(tensor, actual, predictions, loss_fnc, args)

            accum_loss += batch_loss
            batch_loss.backward()
            optimizer.step()


            denominator += num_notes

            if t % args.eval_every == 0:
                val_acc, val_loss = evaluate(model, val_loader, args, loss_fnc)
                trn_acc_arr[t // args.eval_every] = float(tot_corr) / denominator
                # val_acc_arr[t // args.eval_every] = val_acc
                if args.model == "ffnn" or args.model == "rnn":
                    print("Epoch: {:>3d}, Step: {:>7d} | Trn acc: {:.6f} | Trn loss (*1e6): {:.6f} | Val acc: {:.6f} | Val loss (*1e6): {:.6f}"
                            .format(epoch + 1,
                                    t + 1,
                                    trn_acc_arr[t // args.eval_every],
                                    accum_loss / denominator * 1e6,
                                    val_acc,
                                    val_loss * 1e6))
                    if val_acc > best_val_acc:
                        torch.save(model, "./model_dir/model_best_val_acc.pt")
                        best_val_acc = val_acc
                elif args.model == "cnn3d":
                    print("Epoch: {:>4d}, Step: {:>7d} || TRN , VAL || loss (*1e6): {:.6f} , {:.6f} | note acc: {:.6f} , {:.6f} | avg corr notes: {:.6f} , {:.6f} | rest acc: {:.6f} , {:.6f} | length acc: {:.6f} , {:.6f}"
                            .format(epoch,
                                    t,
                                    accum_loss / denominator * 1e6,     val_loss * 1e6,
                                    acc['note_acc'],                    val_acc['note_acc'],
                                    acc['avg_num_corr_notes'],          val_acc['avg_num_corr_notes'],
                                    acc['rest_acc'],                    val_acc['rest_acc'],
                                    acc['length_acc'],                  val_acc['length_acc']





                    ))
                    if val_acc['avg_num_corr_notes'] > best_val_acc:
                        torch.save(model, "./model_dir/model_best_val_acc.pt")
                        best_val_acc = val_acc['avg_num_corr_notes']



                accum_loss = 0
                # mov_avg_corr = 0
                denominator = 0
            t = t + 1
        if epoch % 100 == 0:
            torch.save(model, "./model_dir/{}_epoch_{}_model.pt".format(epoch, args.model))
    #print("Train acc: {}".format(float(tot_corr) / len(trn_loader.dataset)))
    # filename = datetime.datetime.now()
    torch.save(model, "./model_dir/model.pt")
    # os.open("./model_dir/model_info.csv", mode='a')



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


def get_evaluation_stats(tensor, actual, predictions, loss_fnc, args):

    if args.model == "rnn" or args.model == "ffnn":
        predictions = predictions[:, :-1, :]  # remove last prediction, because we don't know the "next note"
    elif args.model == "cnn3d":
        pred_notes, pred_rests, pred_lengths = predictions
        pred_notes = pred_notes[:-1, :, :]
        pred_rests = pred_rests[:-1]
        pred_lengths = pred_lengths[:-1]
        predictions = torch.cat(
            (pred_notes.view((pred_notes.shape[0], -1)), pred_rests.unsqueeze(1), pred_lengths.unsqueeze(1)), dim=1)
        # /TODO comment

    if args.model == "ffnn":
        # remove first `memory` predictions, because the fnn doesn't predict the first `memory` notes
        label = actual[:, args.memory:, :]
    elif args.model == "rnn":
        # remove first prediction, because the rnn doesn't predict the first note
        label = tensor[:, 1:, :]
    elif args.model == "cnn3d":
        label_notes, label_rests, label_lengths = tensor
        label_notes = label_notes.squeeze()
        label_rests = label_rests.squeeze()
        label_lengths = label_lengths.squeeze()

        memory = label_notes.shape[0] - pred_notes.shape[0]  # /
        label_notes = label_notes[memory:, :, :]
        label_rests = label_rests[memory:]
        label_lengths = label_lengths[memory:]
        label = torch.cat(
            (label_notes.view((pred_notes.shape[0], -1)), label_rests.unsqueeze(1), label_lengths.unsqueeze(1)), dim=1)
        # /TODO comment

    batch_loss = loss_fnc(input=predictions, target=label.float())


    if args.model == "ffnn" or args.model == "rnn":
        corr = (binarize_pred_2d(np.array(predictions.detach())[0]) == label)[0]
        corr = np.all(np.array(corr), axis=1)
        acc = float(corr.sum()) / len(corr)
        num_notes = corr.shape[0]  # corr.shape[0] is # notes, so this will calculate the loss per note
    elif args.model == "cnn3d":
        pred_notes = np.array(pred_notes.detach())
        pred_rests = np.array(pred_rests.detach())
        pred_lengths = np.array(pred_lengths.detach())
        pred_notes, pred_rests, pred_lengths = binarize_pred_3d(pred_notes, pred_rests, pred_lengths, args.n_notes)

        label_notes = np.array(label_notes)
        corr_notes = (pred_notes == label_notes)

        num_corr_notes = np.logical_and(corr_notes, pred_notes)
        num_corr_notes = num_corr_notes.reshape((num_corr_notes.shape[0], -1))
        num_corr_notes = np.sum(num_corr_notes, axis=1)

        corr_rests = (pred_rests == label_rests)
        corr_lengths = (pred_lengths == label_lengths)

        corr_notes = corr_notes.reshape((corr_notes.shape[0], -1))
        corr_notes = np.all(np.array(corr_notes), axis=1)


        acc = {
            'note_acc': corr_notes.mean(),
            'avg_num_corr_notes': num_corr_notes.mean(),
            'rest_acc': float(corr_rests.sum())/len(corr_rests),
            'length_acc': float(corr_rests.sum())/len(corr_rests)
        }
        num_notes = corr_notes.shape[0]  # corr.shape[0] is # notes

    return num_notes, batch_loss, acc


def binarize_pred_2d(pred): #/hardcoded rn
    REST_THRESHOLD = 0.8
    is_note = pred[:, 23:24] < REST_THRESHOLD
    pitch = (pred[:,0:12] == np.max(pred[:,0:12], axis=1).reshape(-1,1))
    octave = (pred[:,12:23] == np.max(pred[:,12:23], axis=1).reshape(-1,1))
    pred[:, 0:12] = is_note * pitch
    pred[:, 12:23] = is_note * octave
    pred[:,23:24] =  1 - is_note
    return pred

def binarize_pred_3d(pred_notes, pred_rests, pred_lengths, n_notes):
    # rest_threshold = 0.8 #np.mean(pred_rests) + np.std(pred_rests) * 1.3 # make ~10% of notes rests
    #note_relative_threshold = 0.3

    # is_note = pred_rests < rest_threshold
    pred_rests = np.zeros(pred_rests.shape)
    # TODO implement rests; this assumes rests never happen for now
    note_values = _max_k_last_axis(pred_notes.reshape((pred_notes.shape[0], -1)), n_notes)
    new_pred_notes = np.zeros(pred_notes.shape)
    for i in range(n_notes):
        new_pred_notes = np.logical_or(new_pred_notes, pred_notes == note_values[:, i:(i+1), np.newaxis])

    pred_notes = new_pred_notes
    pred_lengths = np.exp2(np.round(np.log2(pred_lengths)))

    return pred_notes, pred_rests, pred_lengths


def _max_k_last_axis(array, k):
    return np.partition(array, -k)[:,-k:]

def load_model(args):
    if args.model == 'rnn':
        model = RNN(25, args.dim_hidden, 25)
    elif args.model == 'ffnn':
        model = FFNN(25, args.dim_hidden, memory=args.memory, num_hidden_layers=args.num_hidden_layers)
    elif args.model == 'cnn3d':
        model = CNN3D((10,10),((4,4,2),(7,12,2)),10,7,10,7,2,100) #/TODO un hard code
    else:
        raise Exception("Only rnn, ffnn, cnn3d model types currently supported")

    if args.loss_fn == 'mse':
        loss_fn = torch.nn.MSELoss()
    else:
        raise Exception("Only mse loss function currently supported")

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise Exception("Only adam optimizer currently supported")
    return model, loss_fn, optimizer

def evaluate(model, loader, args, loss_fnc):
    if args.model == "rnn" or args.model == "ffnn":
        tot_acc = 0
    elif args.model == "cnn3d":
        tot_acc = {
                'note_acc': 0,
                'avg_num_corr_notes': 0,
                'rest_acc': 0,
                'length_acc': 0
            }
    tot_loss = 0
    tot_batches = 0
    tot_notes = 0
    for i, batch in enumerate(loader):
        tensor = batch['data']
        actual = batch['label']
        predictions = model(tensor)

        num_notes, batch_loss, acc = get_evaluation_stats(tensor, actual, predictions, loss_fnc, args)
        if args.model == "rnn" or args.model == "ffnn":
            tot_acc += acc
        elif args.model == "cnn3d":
            tot_acc = {key: v1 + v2
                       for key, v1, v2 in zip(tot_acc.keys(), tot_acc.values(), acc.values())}
        tot_loss += batch_loss
        tot_batches += 1
        tot_notes += num_notes

    if args.model == "rnn" or args.model == "ffnn":
        acc_per_note = tot_acc / tot_batches
    elif args.model == "cnn3d":
        acc_per_note = {key: v / tot_batches
                        for key, v in zip(tot_acc.keys(), tot_acc.values())}

    loss_per_note = tot_loss / num_notes
    return acc_per_note, loss_per_note


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=4000)
    parser.add_argument('--model', type=str, default="cnn3d")
    parser.add_argument('--emb_dim', type=int, default=100)
    parser.add_argument('--rnn_hidden_dim', type=int, default=100)
    parser.add_argument('--loss_fn', type=str, default="mse")
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--memory', type=int, default=7)
    parser.add_argument('--num_hidden_layers', type=int, default=3)
    parser.add_argument('--dim_hidden', type=int, default=100)
    parser.add_argument('--concat', type=bool, default=True)  # if True, concatenate all pieces together
    parser.add_argument('--first_voice_only', type=bool, default=False)  # if True, only take first Voice
    parser.add_argument('--n_notes', type=int, default=4) # number of simultaneous notes in each chord

    parser.add_argument('--divide_data_pieces', type=int, default=20)  # chop up the midi file tensor into smaller pieces

    parser.add_argument('--overwrite_cached_loaders', type=bool, default=True)  # overwrite the loaders.pkl file

    parser.add_argument('--eval_every', type=int, default=10)

    args = parser.parse_args()

    main(args)