import torch.utils.data as data

class MusicDataset(data.Dataset):

#/will need to add lengths later
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {'data': self.data[index]}

