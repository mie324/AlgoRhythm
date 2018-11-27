import music21 as music
import numpy as np
import random as random

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
def notelist_to_tensor(sorted_notelist):
    atu = most_common_note_length(sorted_notelist) # atomic time unit
    music_length = round((sorted_notelist[-1].offset + sorted_notelist[-1].duration.quarterLength) / atu)
        # length of music in atu's

    pitches_tensor = np.zeros((music_length, len(MIDI_PITCHES)))
    octaves_tensor = np.zeros((music_length, len(MIDI_OCTAVES)))
    lengths_tensor = np.zeros((music_length, 1))
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

        rests_tensor[curr_index:end_index] = 0

        pitches_tensor[curr_index:end_index] = (np.array(MIDI_PITCHES) == note.pitch.pitchClass)
        octaves_tensor[curr_index:end_index] = (np.array(MIDI_OCTAVES) == note.pitch.octave)
        # TODO med priority: try one-hot encoding all 127 notes instead
        prev_note = note.pitch.midi
        prev_index = curr_index
        lengths_tensor[curr_index:end_index] = note.duration.quarterLength

    tensor = np.concatenate((pitches_tensor, octaves_tensor, rests_tensor, lengths_tensor), axis=1)

    return tensor


# convert sorted notelist into tensor, now including length of notes
def notelist_to_tensor_with_length(sorted_notelist):
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

#TODO keep all notes not just top
def notelist_to_tensors_with_3d(sorted_notelist):
    music_length = len(sorted_notelist)
    notes_tensor_3d = np.zeros((music_length, len(MIDI_PITCHES), len(MIDI_OCTAVES)))
    lengths_tensor = np.zeros((music_length,))
    rests_tensor = np.ones((music_length,))
    for i, note in enumerate(sorted_notelist):
        if isinstance(note, music.note.Rest):  # skip if it's a rest
            continue

        rests_tensor[i] = 0  # else make it a rest
        lengths_tensor[i] = note.duration.quarterLength
        notes_tensor_3d[i] = (np.array(MIDI_PITCHES) == note.pitch.pitchClass).reshape((-1,1)) \
                             * (np.array(MIDI_OCTAVES) == note.pitch.octave)

    return notes_tensor_3d, rests_tensor, lengths_tensor

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
def file_to_tensor(path, first_voice_only=False, three_d_tensor=True):
    if three_d_tensor and first_voice_only:
        raise Exception('"3D tensor" and "first voice only" options are not compatible"')
    x = get_stream(path)
    if not first_voice_only:
        x = x.flat
    x = get_firstvoice(x)
    x = firstvoice_to_notechordlist(x)
    x = notechordlist_to_notelist(x)
    x = sorted_notelist(x)
    if three_d_tensor:
        x = notelist_to_tensors_with_3d(x)
    else:
        x = notelist_to_tensor_with_length(x)
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
        if array[23] == 1:  # in same method now bc we need to access multiple rows of the array
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




# takes in 3D tensor, representing notes, and 2 1D tensors, representing rests and length
def convert_array_to_midi_CNN_3D(note_array, rest_array, length_array, path, tempo=120):
    s1 = music.stream.Stream()
    mm = music.tempo.MetronomeMark(number=tempo)  # can set metronome (bpm with beat=quarter note)
    s1.append(mm)
    for i in range(0, note_array.shape[0]):  # goes through array
        note_list = create_note_CNN_3D(note_array[i], rest_array[i], length_array[i])  # appends notes to stream
        if len(note_list) == 1:
            s1.append(note_list[0])
        else:
            chord = music.chord.Chord(note_list)
            s1.append(chord)

    convert_notes_to_midi(s1, path)


# converts a 2D one-hot array into notes, with rest and length encoded as separate variables
# returns note list
def create_note_CNN_3D(s, rest, duration=1):
    pitches = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave_shift = -1
    note_list = []
    if rest == 1:
        n = music.note.Rest()
        n.duration.quarterLength = duration
        note_list.append(n)
        return note_list
    else:
        max_index = np.unravel_index(np.argmax(s, axis=None), s.shape)  # gets indices of max value in 2D array s
        # if multiple max values, gets indices of 1st one
        max_value = s[max_index]

        while max_value > 0:  # while a note still exists, continue appending to note list
            n = music.note.Note()
            n.pitch.name = pitches[max_index[0]]
            n.pitch.octave = max_index[1]+octave_shift
            n.duration.quarterLength = duration
            s[max_index] = 0  # sets array value to 0 to show it was checked
            note_list.append(n)

            max_index = np.unravel_index(np.argmax(s, axis=None), s.shape)  # gets indices of max value in 2D array s
            max_value = s[max_index]  # if no new notes exist, returns a 0
        return note_list

#####################################################################################
# new method that takes ALL notes in a song and converts them into a 3D tensor
def file_to_tensor_flatten(path, first_voice_only=False, three_d_tensor=True):
    if three_d_tensor and first_voice_only:
        raise Exception('"3D tensor" and "first voice only" options are not compatible"')
    x = get_stream(path)
    if not first_voice_only:
        x = x.flat
    x = get_firstvoice(x)

    # General settings assume we can have multiple notes occurring at once due to the flattening
    x = x.getElementsByClass(['Note', 'Chord', 'Rest']).stream()  # obtains the notes, chords, and rests as a stream

    x = notechordlist_to_all_notelist(x)
    x = sorted_notelist(x)  # converts stream to list and sorts list by offset
    if three_d_tensor:
        x = notelist_to_tensors_with_3d_all_notes(x)
    else:
        x = notelist_to_tensor_with_length(x)  # this never actually occurs, but is here for testing
        print("You shouldn't be doing this.")
    return x


def notechordlist_to_all_notelist(notechordlist):
    length = len(notechordlist)  # length is the length of original list, minus whatever we cut later
    # for i in range(0, len(notechordlist)):  # loops through list
    i = 0
    while i < length:  # loops through list
        element = notechordlist[i]
        if isinstance(element, music.chord.Chord):  # if element is a chord
            offset = element.offset
            for note in element:  # obtain the notes in chord element
                note.offset = note.offset+offset  # obtain new offset
                notechordlist.append(note)  # adds new notes to end of stream (i think end, doesn't specify)
            notechordlist.pop[i]  # remove chord
            i = i-1
            length = length-1
        i = i+1

            # offset = element.offset  # get old offset
            # freq = element.sortFrequencyAscending()  # sort chord by freq asc
            # highNote = freq[-1]  # get last note(highest)
            # highNote.offset = highNote.offset + offset  # change offset to be chord's offset
            # notechordlist[i] = highNote  # replace the chord with the note

    return notechordlist


def notelist_to_tensors_with_3d_all_notes(sorted_notelist):
    music_length = len(sorted_notelist)

    notes_tensor_3d = np.zeros((1, len(MIDI_PITCHES), len(MIDI_OCTAVES)))
    lengths_tensor = []
    rests_tensor = np.ones((music_length,))
    prev_offset = 0  # stores the offset of the previous note, start at 0
    j = 0
    notes_in_slice = []  # stores the notes for a particular slice

    # each slice of notes_tensor_3d represents a time-step with unchanging notes
    for i, note in enumerate(sorted_notelist):
        if note.offset > prev_offset:  # if new time step
            difference = note.offset-prev_offset
            lengths_tensor.append = difference  # cut off length of prev timestep
            # add new time slice
            np.append(notes_tensor_3d, np.zeros((1, len(MIDI_PITCHES), len(MIDI_OCTAVES))), axis=0)
            j = j + 1  # move time index
            prev_offset = note.offset  # start new time step

            # check for values w durations <= 0, remove said values from list
            for k in range(notes_in_slice-1, -1, -1):
                notes_in_slice.duration.quarterLength = notes_in_slice.duration.quarterLength-difference
                if notes_in_slice.duration.quarterLength <= 0:
                    notes_in_slice.pop(k)  # chuck notes that don't have enough duration
                else:
                    # repeat notes that still have duration
                    note = notes_in_slice[k]
                    notes_tensor_3d[j, note.pitch.pitchClass, note.pitch.octave + 1] = 1

        if i == len(sorted_notelist):  # if last note in list, end off everything
            lengths_tensor.append = note.duration.quarterLength

        if isinstance(note, music.note.Rest):  # skip if it's a rest
            continue

        rests_tensor[j] = 0  # else make it a note
        notes_in_slice.append(note)  # add info to lists
        notes_tensor_3d[j, note.pitch.pitchClass, note.pitch.octave + 1] = 1  # sets a value in the 2d slice to 1
        # +1 in octave bc octaves start at -1

    # cut short rest tensor to cut off lingering one values
    return notes_tensor_3d, rests_tensor[:len(lengths_tensor)], np.array(lengths_tensor)

########################################################################################

def voice_test():  # to test and see what voices are available in a song
    # path = "./midi/bach_wtc1/Prelude3.mid"
    # s = get_stream(path)
    # # s.show('text')
    # f = s.flat
    # f.show('text')
    s = music.stream.Stream()
    s.append(music.note.Note('C'))
    s.append(music.note.Note('D'))
    print("Trial Complete")


if __name__ == '__main__':
    voice_test()
    # path1 = "./midi/bach_minuet.mid"
    # path2 = "./midi/bach_wtc1/Prelude1.mid"
    # a = file_to_tensor(path1)
    # b = file_to_tensor(path2)
    # c = file_to_tensor(path1, False)
    # d = file_to_tensor(path2, False)
    #
    # s = get_stream(path1)
    # # s = get_flattened(s)
    # # s_string = convert_to_string(s)
    # # print(s_string)
    # #s.show('text')  # shows everything in a stream, useful for debugging
    # s_sounds = s.getElementsByClass(['Note','Chord','Rest']).stream()
    # p = s.parts.stream()
    # p[0].voices[0].show('text')
    # print("Trial Complete")

