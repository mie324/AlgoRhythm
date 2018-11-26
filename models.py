import torch
import torch.nn as nn
import torch.nn.functional as F
from midi_converter import MIDI_PITCHES, MIDI_OCTAVES

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
    def __init__(self, dim_music, dim_hidden, memory=3, num_hidden_layers=5):
        super(FFNN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        # need to use a ModuleList instead of a regular list, otherwise the backpropagation won't take these layers into account
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

class CNN3D(nn.Module):
    def __init__(self, n_channels_list, kernel_size_list, rest_n_channels, rest_kernel_size, length_n_channels,
                 length_kernel_size, n_fc_hidden_layers, fc_layer_size): #kernel_size_list should be a tuple of tuples
        super(CNN3D, self).__init__()

        self.n_channels_list = n_channels_list
        self.kernel_size_list = kernel_size_list
        self.rest_kernel_size = rest_kernel_size
        self.length_kernel_size = length_kernel_size

        # find largest time-size of all the kernels
        kernel_time_sizes = [kernel_size_tuple[0] for kernel_size_tuple in kernel_size_list]
        kernel_time_sizes.extend([rest_kernel_size, length_kernel_size])
        self.max_kernel_time_size = max(kernel_time_sizes)

        self.conv_layers = nn.ModuleList()
        for n_channels, kernel_size_tuple in zip(n_channels_list, kernel_size_list):
            self.conv_layers.append(nn.Conv3d(1, n_channels, kernel_size_tuple))
        self.conv_rest = nn.Conv1d(1, rest_n_channels, rest_kernel_size)
        self.conv_length = nn.Conv1d(1, length_n_channels, length_kernel_size)

        # fc_dim_in = self.n_channels_list[-1] *
        fc_dim_in = 3820 #TODO un hard code
        self.fc_in = nn.Linear(fc_dim_in, fc_layer_size)

        self.fc_layers = nn.ModuleList()
        for i in range(n_fc_hidden_layers):
            self.fc_layers.append(nn.Linear(fc_layer_size, fc_layer_size))

        self.fc_out = nn.Linear(fc_layer_size, len(MIDI_PITCHES) * len(MIDI_OCTAVES) + 2)

    def forward(self, input):
        # for the CNN3D, input is a tuple consisting of (the 3d tensor, rest tensor, length tensor)
        x, r, l = input
        for i in range(len(self.conv_layers)):
            x2 = CNN3D.wraparound_pitch(x, dist=(self.kernel_size_list[i][1] - 1)) # dist = size of kernel in pitch dimension minus one
            x2 = x2.float()
            x2 = x2.unsqueeze(0)
            x2 = self.conv_layers[i](x2)

            # flatten all the outputs of all the conv layers in all dimensions except time
            # and concatenate them, in preparation for feeding into the fcnet (fully connected net)
            #/ maybe need to swap dims???
            x2 = x2.squeeze()
            x2 = x2.transpose(0,1).contiguous() #/
            x2 = x2.view((x2.shape[0], -1))
            x2 = x2[(self.max_kernel_time_size - self.kernel_size_list[i][0]):, :] #TODO add comment
            if i == 0:
                tensor_into_fcnet = x2
            else:
                tensor_into_fcnet = torch.cat((tensor_into_fcnet, x2), dim=1)

        r = r.unsqueeze(0).float()
        r = self.conv_rest(r)
        r = r.squeeze()
        r = r.transpose(0, 1).contiguous()
        r = r.view((r.shape[0], -1))
        r = r[(self.max_kernel_time_size - self.rest_kernel_size):, :]  # TODO add comment
        tensor_into_fcnet = torch.cat((tensor_into_fcnet, r), dim=1)


        l = l.unsqueeze(0).float()
        l = self.conv_length(l)
        l = l.squeeze()
        l = l.transpose(0, 1).contiguous()
        l = l.view((l.shape[0], -1))
        l = l[(self.max_kernel_time_size - self.rest_kernel_size):, :]  # TODO add comment
        tensor_into_fcnet = torch.cat((tensor_into_fcnet, l), dim=1)

        tensor_into_fcnet = F.leaky_relu(tensor_into_fcnet)

        tensor_into_fcnet = F.leaky_relu(self.fc_in(tensor_into_fcnet))
        for i in range(len(self.fc_layers)):
            tensor_into_fcnet = F.leaky_relu(self.fc_layers[i](tensor_into_fcnet))
        tensor_into_fcnet = F.sigmoid(self.fc_out(tensor_into_fcnet)) #TODO change to remove warning
        output_rest_tensor = tensor_into_fcnet[:, 0]
        output_length_tensor = tensor_into_fcnet[:, 1]
        output_notes_tensor = tensor_into_fcnet[:, 2:]

        output_notes_tensor = output_notes_tensor.view((-1,len(MIDI_PITCHES), len(MIDI_OCTAVES)))
        return output_notes_tensor, output_rest_tensor, output_length_tensor


    @staticmethod
    def wraparound_pitch(tensor, dist=None):
        # assuming pitch dimension is 2
        if dist is None:
             dist = tensor.shape[2] - 1
        return torch.cat((tensor[:,:,-dist:,:],tensor,tensor[:,:,:dist,:]), dim=2)

