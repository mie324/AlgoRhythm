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



# class CNN(nn.Module):
#     def __init__(self, embedding_dim, vocab, n_filters, filter_sizes):
#         super(CNN, self).__init__()
#
#         ######
#
#         # 4.3 YOUR CODE HERE
#         self.embed = nn.Embedding.from_pretrained(vocab.vectors)
#         self.conv_a = nn.Conv2d(1, n_filters, (filter_sizes[0], embedding_dim))
#         self.conv_b = nn.Conv2d(1, n_filters, (filter_sizes[1], embedding_dim))
#
#         self.fc = nn.Linear(embedding_dim, 1)
#
#         ######
#
#
#     def forward(self, x, lengths=None):
#         ######
#
#         # 4.3 YOUR CODE HERE
#         x = self.embed(x)
#         x = x.reshape(-1, 1, x.shape[1], x.shape[2])
#         x = x.permute(2,1,0,3)
#
#         a = F.relu(self.conv_a(x)).squeeze(dim=3)
#         b = F.relu(self.conv_b(x)).squeeze(dim=3)
#         a, _ = torch.max(a, 2)
#         b, _ = torch.max(b, 2)
#
#         x = torch.cat((a, b), dim=1)
#         x = self.fc(x)
#         x = F.sigmoid(x)
#         x = x.view(-1)
#         ######
#         return x
