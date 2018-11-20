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
