import os

from torch.utils.data.dataloader import DataLoader

DATASET_ROOT = '“Path to store the dataset”'

class Path(object):
    @staticmethod
    def getPath(dataset):
        path = os.path.join(DATASET_ROOT, 'Brats2018')
        # path = os.path.join(DATASET_ROOT, 'Brats2020')

        return os.path.realpath(path)
