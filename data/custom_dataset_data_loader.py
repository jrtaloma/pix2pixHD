import numpy as np
import torch.utils.data
from torch.utils.data import WeightedRandomSampler
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        # Default data loader
        if not opt.isTrain or not opt.use_weighted_random_sampler:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                shuffle=not opt.serial_batches,
                num_workers=int(opt.nThreads))
        # Data loader with weighted random sampler
        else:
            labels = np.array(self.dataset.labels).astype(float)
            _, counts = np.unique(labels, return_counts=True)
            # |negatives| > |positives|
            if counts[0] > counts[1]:
                for i in range(len(labels)):
                    labels[i] = counts[0]/counts[1] if labels[i] == 1 else 1
            # |positives| > |negatives|
            else:
                for i in range(len(labels)):
                    labels[i] = counts[1]/counts[0] if labels[i] == 0 else 1
            sampler = WeightedRandomSampler(labels, len(labels), replacement=True)
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                sampler=sampler,
                num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
