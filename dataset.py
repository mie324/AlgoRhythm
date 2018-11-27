import torch.utils.data as data

class MusicDataset(data.Dataset):

#/will need to add lengths later
    def __init__(self, data, label=None):
        self.data = data
        if label is None:
            self.label = self.data
        else:
            self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {'data': self.data[index], 'label' : self.label[index]}

