import torch
from torch.utils.data import Dataset


class RNNDataset(Dataset):
    def __init__(self, train, lens, labels, dense_datasets_holder):
        self.train = train
        self.labels = labels
        self.lens = lens
        self.len = len(train)
        self.dense_datasets_holder = dense_datasets_holder

        if labels is None:
            self.test = True
        else:
            self.test = False

    def __getitem__(self, idx):
        train = self.train[idx]
        items_lookup = self.dense_datasets_holder.items[train[:, 0:1].squeeze(1).long()]
        items_categories_lookup = self.dense_datasets_holder.items_categories[train[:, 1:2].squeeze(1).long()]
        shops_lookup = self.dense_datasets_holder.shops[train[:, 2:3].squeeze(1).long()]

        mixed = (items_lookup + items_categories_lookup + shops_lookup) / 3.0

        train = torch.cat([train, items_lookup, items_categories_lookup, shops_lookup], dim=1)

        train = torch.cat([train, mixed], dim=1)

        if self.test:
            labels = None
        else:
            labels = self.labels[idx]

        return train, self.lens[idx], labels

    def __len__(self):
        return self.len
