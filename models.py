import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers=5):
        super(RNN, self).__init__()
        # self.gru = nn.GRU(input_size=dim_in, hidden_size=dim_hidden, num_layers=num_layers)
        self.lstm = nn.LSTM(input_size=dim_in, hidden_size=dim_hidden, num_layers=4)
        self.fc1 = nn.Linear(dim_hidden, dim_hidden//2)
        self.fc2 = nn.Linear(dim_hidden//2, dim_hidden//2)
        self.out = nn.Linear(dim_hidden // 2, dim_out)

    def forward(self, x): #/will later need lengths as a parameter too if not concatenating all training examples
        #x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=False)#/batch_first might =True
        x, hidden = self.lstm(x.float())
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.out(x)
        x = F.sigmoid(x)
        return x


class FFNN(nn.Module):
    #/ TODO to try: do a convolutional thing over pitches (wraparound) and over octaves (no wraparound)
    def __init__(self, dim_music, dim_hidden, memory=3, num_hidden_layers=5):
        super(FFNN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.layer_in = nn.Linear(dim_music * memory, dim_hidden)
        for i in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(dim_hidden, dim_hidden))
        self.layer_out = nn.Linear(dim_hidden, dim_music)

    def forward(self, x):
        x = x.float()
        x = F.leaky_relu(self.layer_in(x))
        for layer in self.hidden_layers:
            x = F.leaky_relu(layer(x))
        x = F.sigmoid(self.layer_out(x))
        return x


# work in progress
class CNN(nn.Module):
    def __init__(self, convolve_time, n_filters, filter_sizes):
        super(CNN, self).__init__()

        ######

        # Create 2D convolutions over note data and time
        self.conv_pitch = nn.Conv2d(1, n_filters, (filter_sizes[0], convolve_time))  # convolve over pitch
        self.conv_octave = nn.Conv2d(1, n_filters, (filter_sizes[1], convolve_time))  # convolve over octave
        self.conv_rest = nn.Conv1d(1, n_filters, convolve_time)

        self.fc = nn.Linear(convolve_time, 1)
        self.sigmoid = nn.Sigmoid()
        ######

    def forward(self, x):  # x is input
        ######

        # assume you get a batch's worth set of notes, width 24 (pitch+octave+rest), and some length (3-d array)
        x = x.reshape(-1, 1, x.shape[1], x.shape[2])
        x = x.permute(2,1,0,3)

        # assume dimensions are (batch num, 1, notes, time). Take convolutions over pitch, octave, and rest
        x_pitch = F.relu(self.conv_pitch(x[:, :, 0:12, :])).squeeze()
        x_octave = F.relu(self.conv_octave(x[:, :, 12:23, :])).squeeze()
        x_rest = F.relu(self.conv_rest(x)).squeeze(x[:, :, 23, :])

        a, _ = torch.max(a, 2)
        b, _ = torch.max(b, 2)

        x = torch.cat((a, b), dim=1)
        x = self.fc(x)
        x = self.sigmoid(x)
        x = x.view(-1)
        ######
        return x


# Attempt to convolve over pitch, octave, rests and time
class CNN_v2(nn.Module):  # take in a series of 32 note sequences, predict next note in sequence
    def __init__(self, memory=32, convolve_time=[4, 8, 16], n_filters=8, hidden_dim=100, hidden_layers=0):
        super(CNN_v2, self).__init__()

        ######

        # tensor structure consists of 2d array of points
        self.conv_note1 = nn.Conv2d(1, n_filters, [12, convolve_time[0]])
        self.conv_rest1 = nn.Conv1d(1, n_filters, convolve_time[0])
        self.conv_length1 = nn.Conv1d(1, n_filters, convolve_time[0])

        self.conv_note2 = nn.Conv2d(1, n_filters, [12, convolve_time[1]])
        self.conv_rest2 = nn.Conv1d(1, n_filters, convolve_time[1])
        self.conv_length2 = nn.Conv1d(1, n_filters, convolve_time[1])

        self.conv_note3 = nn.Conv2d(1, n_filters, [12, convolve_time[2]])
        self.conv_rest3 = nn.Conv1d(1, n_filters, convolve_time[2])
        self.conv_length3 = nn.Conv1d(1, n_filters, convolve_time[2])

        # plug into linear layers after
        self.hidden_layers = nn.ModuleList()
        self.linear_size = n_filters*6*123
        self.fc_start = nn.Linear(self.linear_size, hidden_dim)
        for i in range(hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.fc_final = nn.Linear(hidden_dim, 134)
        # returns a single note: 132 values for note (pitch and octave), 1 value for rest, 1 for length
        self.sigmoid = nn.Sigmoid()
        ######

    def forward(self, x):  # x is input
        ######
        #print("OG Shape")
        #print(x.shape)
        # assume you get a batch's worth set of notes, width 134 (note+rest+length), and some length (3-d array)
        #x = x.reshape(-1, 1, x.shape[1], x.shape[2])
        #x = x.permute(2, 1, 0, 3)
        x = x.permute(1, 0, 2)
        x = x.reshape(x.shape[0], x.shape[1], 134, -1)
        #print("converted shape")
        #print(x.shape)
        x = x.float()

        # assume dimensions are (time/batches, 1, notes, memory). Take convolutions over notes/length/rest & time
        x_note1 = F.relu(self.conv_note1(x[:, :, 0:132, :]))
        x_rest1 = F.relu(self.conv_rest1(x[:, :, 132, :]))
        x_length1 = F.relu(self.conv_length1(x[:, :, 133, :]))

        x_note2 = F.relu(self.conv_note2(x[:, :, 0:132, :]))
        x_rest2 = F.relu(self.conv_rest2(x[:, :, 132, :]))
        x_length2 = F.relu(self.conv_length2(x[:, :, 133, :]))

        #x_note3 = F.relu(self.conv_note3(x[:, :, 0:132, :])).squeeze()
        #x_rest3 = F.relu(self.conv_rest3(x[:, :, 132, :])).squeeze()
        #x_length3 = F.relu(self.conv_length3(x[:, :, 133, :])).squeeze()
        x_note = torch.cat((x_note1, x_note2), dim=3)
        x_rest = torch.cat((x_rest1, x_rest2), dim=2)
        x_length = torch.cat((x_length1, x_length2), dim=2)
        x_note = x_note.permute(0, 1, 3, 2)
        x_rest = x_rest.unsqueeze(3)
        x_length = x_length.unsqueeze(3)
        #print(x_note.shape)
        #print(x_rest.shape)
        #print(x_length.shape)
        x = torch.cat((x_note, x_rest, x_length), dim=3)
        #print(x.shape)
        x = x.view(-1, self.linear_size)

        x = F.leaky_relu(self.fc_start(x))
        for layer in self.hidden_layers:
            x = F.leaky_relu(layer(x))
        x = self.sigmoid(self.fc_final(x))
        #print(x.shape)

        ######
        return x


def pad_circular(x, pad=11):  # x = array to pad, pad = size of padding
    # This method is designed to apply wraparound padding to 1st dimension of x (pitch)
    x = torch.cat([x, x[0:pad]], dim=0)  # appends 1st pad layers onto right side of x
    #x = torch.cat([x[-2 * pad:-pad], x], dim=0)  # from online
    x = torch.cat([x[-pad:], x], dim=0)  # appends last pad layers onto left side of x

    return x

