import torch.utils.data as data

class MusicDataset(data.Dataset):

#/will need to add lengths later
    def __init__(self, data, data_with_shifted):
        self.data = data
        self.data_with_shifted = data_with_shifted

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {'data': self.data[index], 'data_with_shifted' : self.data_with_shifted[index]}

