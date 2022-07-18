import torch
from torch.utils.data import DataLoader

# dataset = DatasetGenerator(data_dict)
# dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=drop_last, collate_fn=collate_fn_padd)


class DatasetGenerator:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item][3]
        label = self.dataset[item][2]
        return torch.tensor(data), torch.tensor(int(label))


def pad_tensor(vec, pad):
    pad_size = list(vec.shape)
    pad_size[0] = pad - vec.size(0)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=0)


def collate_fn_padd(batch):
    # pad according to max_len
    audio_max_len = max(map(lambda x: x[0].shape[0], batch))
    data, labels, lens = [], [], []
    for idx in range(len(batch)):
        data.append(pad_tensor(batch[idx][0], pad=audio_max_len))
        labels.append(batch[idx][1])
        lens.append(len(batch[idx][0]))
    data, labels, lens = (
        torch.stack(data, dim=0),
        torch.stack(labels, dim=0),
        torch.tensor(lens),
    )
    return data, labels, lens
