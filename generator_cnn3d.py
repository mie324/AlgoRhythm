import numpy as np

from models import *
from midi_converter import MIDI_PITCHES, MIDI_OCTAVES, convert_array_to_midi_CNN_3D
from main import binarize_pred_3d

def chord(pitches, octaves):
    n = np.zeros((1, 1, len(MIDI_PITCHES), len(MIDI_OCTAVES)))
    for pitch, octave in zip(pitches, octaves):
        n[0, 0, pitch, octave] = 1
    return n

if __name__ == "__main__":
    # pitches of notes for convenience
    c=0;db=1;d=2;eb=3;e=4;f=5;gb=6;g=7;ab=8;a=9;bb=10;b=11

    epoch = 600
    n_notes_list = [1,2,3,4]
    model = torch.load("./model_dir/{}_epoch_cnn3d_model.pt".format(epoch))

    for n_notes in n_notes_list:
        # create an initializing sequence consisting of the chords (Cmaj, Fmaj, Gmaj, Cmaj) x2
        notes = np.concatenate(2*(chord((c,e,g),(5,5,5)), chord((c,f,a),(5,5,5)), chord((d,g,b),(5,5,5)), chord((c,e,g),(5,5,5))), axis=1)
        rests = np.zeros((1,8))
        lengths = np.ones((1,8))
        print("", end="")

        for i in range(50):
            predictions = model((torch.Tensor(notes), torch.Tensor(rests), torch.Tensor(lengths)))

            # get predicted notes (3d array), rests (1d) and lengths (1d)
            pred_notes, pred_rests, pred_lengths = predictions
            pred_notes = np.array(pred_notes.detach())
            pred_rests = np.array(pred_rests.detach())
            pred_lengths = np.array(pred_lengths.detach())

            # binarize
            pred_notes, pred_rests, pred_lengths = binarize_pred_3d(pred_notes, pred_rests, pred_lengths, n_notes)

            # append latest prediction
            notes = np.concatenate((notes, pred_notes[np.newaxis,-1:,:,:]), axis=1)
            rests = np.concatenate((rests, pred_rests[np.newaxis,-1:]), axis=1)
            lengths = np.concatenate((lengths, pred_lengths[np.newaxis,-1:]), axis=1)

        # Removes all but one note/chord of the initial starting sequence
        notes = notes[:, 7:, :, :]
        rests = rests[:, 7:]
        lengths = lengths[:, 7:]

        # Prints the music tensors as midi
        convert_array_to_midi_CNN_3D(notes.squeeze(0), rests.squeeze(0), lengths.squeeze(0), "./generated_midi/generated_cnn3d_{}_epochs_{}_notes_new.mid".format(epoch, n_notes), tempo=40)

        print("Complete")
