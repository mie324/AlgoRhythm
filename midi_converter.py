import music21 as music
import numpy as np
import random as random
from torchtext.data.dataset import Dataset
from torchtext.data.example import Example
from torchtext.data.iterator import BucketIterator
from dataset import MusicDataset

VERBOSE = True

MIDI_PITCHES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # C=0 through B=11
MIDI_OCTAVES = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # octaves change number between B and C, and middle C is C4


# returns score based on name
def get_stream(path):
    score = None
    try:
        score = music.converter.parseFile(path)
    except music.midi.MidiException:
        raise Exception("Midi file could not be parsed: {}".format(path))
    return score


# flat ensures that all elements are 1 one stream (ie no nested Music21Objects)
#/Jenn: do we really need a function for one-liners?
def get_flattened(stream):
    return stream.flat


# only works for notes?
def convert_to_string(s):
    s_string = ""
    for myElement in s:
        s_string += "%s: %s; " % (myElement.offset, myElement.name)
    return s_string


# returns multiple streams based on instrument
def multi_stream(s):
    parts = music.instrument.partitionByInstrument(s)
    return parts

# returns notes, chords, and other objects from 1st voice of 1st part
def get_firstvoice(score):
    parts = score.getElementsByClass(music.stream.Part)  # obtains parts
    if len(parts) > 0:  # check if parts exist
        score = parts[0]  # if so, take the 1st part, else skip and search for voices
    voices = score.getElementsByClass(music.stream.Voice)  # repeat but for voices
    if len(voices) > 0:
        score = voices[0]

    return score  # return 1st voice of 1st part


# eliminates unnecessary data from the Voice object, such as MetronomeMark's
def firstvoice_to_notechordlist(firstvoice):
    return list(firstvoice.getElementsByClass(['Note', 'Chord', 'Rest']).stream())

# goes through a stream and converts chords to notes
def notechordlist_to_notelist(notechordlist):
    for i in range(0, len(notechordlist)):  # loops through list
        element = notechordlist[i]
        if isinstance(element, music.chord.Chord):  # if element is a chord
            offset = element.offset  # get old offset
            freq = element.sortFrequencyAscending()  # sort chord by freq asc
            highNote = freq[-1]  # get last note(highest)
            highNote.offset = highNote.offset + offset  # change offset to be chord's offset
            notechordlist[i] = highNote  # replace the chord with the note

    return notechordlist


# converts Voice to List of Notes, if it isn't list already.
# Then sort by the notes' starting times, if it isn't sorted already
def sorted_notelist(notelist):
    notelist = list(notelist)
    return sorted(notelist, key=lambda x: x.offset)


# converts a sorted List of Notes into a one-hot-encoded Tensor (np array).
# One-hot format: pitch = NUM_PITCHES bits, octave = NUM_OCTAVES bits
# Assumption: exactly 0 or 1 notes are playing at any given time.
#     If the List contains more than 1 playing simultaneously, the latest-starting one takes precedence.
def notelist_to_tensor(sorted_notelist, has_rest_col=True):
    atu = most_common_note_length(sorted_notelist) # atomic time unit
    music_length = round((sorted_notelist[-1].offset + sorted_notelist[-1].duration.quarterLength) / atu)
        # length of music in atu's

    pitches_tensor = np.zeros((music_length, len(MIDI_PITCHES)))
    octaves_tensor = np.zeros((music_length, len(MIDI_OCTAVES)))
    lengths_tensor = np.zeros((music_length, 1))
    if has_rest_col:
        rests_tensor = np.ones((music_length, 1))

    prev_index = -1
    prev_note = -1
    for note in sorted_notelist:
        curr_index = round(note.offset / atu)  # (inclusive)
        end_index = curr_index + round(note.duration.quarterLength / atu)  # (exclusive)

        if isinstance(note, music.note.Rest):
            continue

        # If >=2 notes/rests at same time, only take highest one and ignore rests and lower notes
        if curr_index == prev_index and note.pitch.midi <= prev_note:
            continue

        if has_rest_col:
            rests_tensor[curr_index:end_index] = 0

        pitches_tensor[curr_index:end_index] = (np.array(MIDI_PITCHES) == note.pitch.pitchClass)
        octaves_tensor[curr_index:end_index] = (np.array(MIDI_OCTAVES) == note.pitch.octave)
        # TODO med priority: try one-hot encoding all 127 notes instead
        prev_note = note.pitch.midi
        prev_index = curr_index
        lengths_tensor[curr_index:end_index] = note.duration.quarterLength

    if has_rest_col:
        tensor = np.concatenate((pitches_tensor, octaves_tensor, rests_tensor, lengths_tensor), axis=1)
    else:
        tensor = np.concatenate((pitches_tensor, octaves_tensor), axis=1)
    return tensor


# convert sorted notelist into tensor, now including length of notes
def notelist_to_tensor2(sorted_notelist, has_rest_col=True):
    music_length = len(sorted_notelist)
    pitches_tensor = np.zeros((music_length, len(MIDI_PITCHES)))
    octaves_tensor = np.zeros((music_length, len(MIDI_OCTAVES)))
    lengths_tensor = np.zeros((music_length, 1))
    rests_tensor = np.ones((music_length, 1))

    for num, note in enumerate(sorted_notelist):
        if isinstance(note, music.note.Rest):  # skip if it's a rest
            continue

        rests_tensor[num] = 0  # else make it a rest
        pitches_tensor[num] = (np.array(MIDI_PITCHES) == note.pitch.pitchClass)
        octaves_tensor[num] = (np.array(MIDI_OCTAVES) == note.pitch.octave)
        lengths_tensor[num] = note.duration.quarterLength

    tensor = np.concatenate((pitches_tensor, octaves_tensor, rests_tensor, lengths_tensor), axis=1)
    return tensor



# finds the most common note length of a sorted notelist, to be used as an atomic time unit.
def most_common_note_length(sorted_notelist):
    freq = {}
    for i in range(len(sorted_notelist) - 1):
        note_length = sorted_notelist[i + 1].offset - sorted_notelist[i].offset
        if note_length != 0.0: # may have problem with floating point comparison
            if note_length not in freq.keys():
                freq[note_length] = 0
            freq[note_length] += 1

    max_freq = -1
    most_common_length = -1

    for note_length in sorted(freq.keys(), reverse=True): #get most frequent note length
        if freq[note_length] >= max_freq:
            most_common_length = note_length
            max_freq = freq[note_length]
    return most_common_length


# goes from midi filepath to tensor, combining all the previous helper functions.
def file_to_tensor(path, first_voice_only=False, has_rest_col=True, has_length_col=True):
    x = get_stream(path)
    if not first_voice_only:
        x = x.flat
    x = get_firstvoice(x)
    x = firstvoice_to_notechordlist(x)
    x = notechordlist_to_notelist(x)
    x = sorted_notelist(x)
    if has_length_col:
        x = notelist_to_tensor2(x, has_rest_col=has_rest_col)
    else:
        x = notelist_to_tensor(x, has_rest_col=has_rest_col)
    return x

# # DOESN'T WORK
# # goes from midi filepaths to torchtext.data.BucketIterator
# def files_to_bucketiterator(pathlist, batch_size, first_voice_only=False, has_rest_col=True):
#     tensorlist = []
#     for path in pathlist:
#         if VERBOSE:
#             print("Processing {} ...".format(path))
#         t = file_to_tensor(path, first_voice_only=first_voice_only, has_rest_col=has_rest_col)
#         tensorlist.append(t)
#     dataset = Dataset(tensorlist, {"music" : "Music"})
#
#     bi = BucketIterator(dataset, batch_size, sort_key=lambda x: x.shape[0], sort_within_batch=True, repeat=False)
#     if VERBOSE:
#         print("Processing complete.")
#     return bi
#
#
# # DOESN'T WORK
# # goes from midi filepaths to MusicDataset (defined in dataset.py) then to torch.utils.data.DataLoader
# def files_to_dataloader(pathlist, batch_size, first_voice_only=False, has_rest_col=True, shuffle=True):
#     tensorlist = []
#     lengths = []
#     for path in pathlist:
#         if VERBOSE:
#             print("Processing {} ...".format(path))
#         t = file_to_tensor(path, first_voice_only=first_voice_only, has_rest_col=has_rest_col)
#         tensorlist.append(t)
#         lengths.append(len(t))
#     dataset = MusicDataset(tensorlist, lengths)
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#     return data_loader



# # converts from a batch (coming from the DataLoader) (which will be a list of 2D tensors of varying sizes)
# # to a 3D tensor where the original 2D tensors are padded with zeros until the maximum length.
# def batch_to_tensor(data, lengths):
#     num_feats = data[0].shape[1]
#     tensor = np.zeros((len(data), max(lengths), num_feats))
#     for i in range(len(data)):
#         tensor[i, 0:lengths[i],:] = data[i]
#     return tensor, lengths




def convert_notes_to_midi(notes, path):  # converts stream of notes into file
    notes.write('midi', fp=path)


def convert_array_to_midi(array, path):  # takes in 2d array. converts into file
    s1 = music.stream.Stream()
    for i in range(0, array.shape[0]):  # goes through array
        s1.append(create_note(array[i]))  # appends notes to stream

    convert_notes_to_midi(s1, path)

def convert_array_to_midi2(array, path, tempo=120):
    s1 = music.stream.Stream()
    s1.append(music.tempo.MetronomeMark(number=tempo))  #adds tempo

    # rounds durations to nearest power of 2
    array[:, 24] = np.exp2(np.round(np.log2(array[:, 24])))

    # finds number of beats each note is, rounded to nearest 1/4
    mean_beat = np.mean(array[:, 24])  # avg duration of beats
    #array[:, 24] = array[:, 24]/mean_beat
    #array[:, 24] = np.round(array[:, 24]*2)
    #array[:, 24] = array[:, 24]/2

    for i in range(0, array.shape[0]):  # goes through array
        s1.append(create_note(array[i], array[i, 24]))  # appends notes to stream, adds in length

    convert_notes_to_midi(s1, path)

# v2 method: takes in 2d array, converts into file.  WORK IN PROGRESS
def convert_array_to_midi_alter(array, path, tempo=120):
    # new features: tempo alterable, able to change note duration.
    s1 = music.stream.Stream()
    mm = music.tempo.MetronomeMark(number=tempo)  # can set metronome (i assume bpm? with beat=quarter note)
    s1.append(mm)

    pitches = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave_shift = -1

    extend_chance = 0.6  # currently we use random seed to determine if notes should be extended or not
    seed = 1
    random.seed(seed)

    song_length = array.shape[0]  # length of output array
    for i in range(0, song_length):  # goes through array
        if s[23] == 1:  # in same method now bc we need to access multiple rows of the array
            n = music.note.Rest()
        else:
            index_pitch = int(np.argmax(array[i, 0:12]))  # find which pitch from one-hot
            index_octave = int(np.argmax(array[i, 12:23]))  # find which octave from one-hot

            n = music.note.Note()
            n.pitch.name = pitches[index_pitch]
            n.pitch.octave = index_octave + octave_shift
            j = i+1
            duration = 1
            while j < song_length:
                # Find next note's pitch and octave
                next_pitch = int(np.argmax(array[j, 0:12]))
                next_octave = int(np.argmax(array[j, 12:23]))

                # If next note matches current note, we have a chance to extend the note
                if next_octave == index_octave and next_pitch == index_pitch:
                    if random.random() < extend_chance:
                        duration += 1  # increase duration of note
                        j += 1
                        i += 1  # can we increase enumerator?
                    else:
                        j = song_length
                else:
                    j = song_length
            n.duration.quarterLength = duration  # set duration of note (in quarter notes)
        s1.append(n)

    convert_notes_to_midi(s1, path)


def create_note(s, duration=1):  # converts a numpy list of one-hot to a note or rest
    pitches = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave_shift = -1
    if s[23] == 1:
        n = music.note.Rest()
        n.duration.quarterLength = duration
        return n
    else:
        index_pitch = int(np.argmax(s[0:12]))  # find which pitch from one-hot
        index_octave = int(np.argmax(s[12:23]))  # find which octave from one-hot
        n = music.note.Note()
        n.pitch.name = pitches[index_pitch]
        n.pitch.octave = index_octave+octave_shift
        n.duration.quarterLength = duration
        return n




if __name__ == '__main__':
    path1 = "./midi/bach_minuet.mid"
    path2 = "./midi/bach_wtc1/Prelude1.mid"
    a = file_to_tensor(path1)
    b = file_to_tensor(path2)
    c = file_to_tensor(path1, False)
    d = file_to_tensor(path2, False)

    s = get_stream(path1)
    # s = get_flattened(s)
    # s_string = convert_to_string(s)
    # print(s_string)
    #s.show('text')  # shows everything in a stream, useful for debugging
    s_sounds = s.getElementsByClass(['Note','Chord','Rest']).stream()
    p = s.parts.stream()
    p[0].voices[0].show('text')
    print("Trial Complete")

