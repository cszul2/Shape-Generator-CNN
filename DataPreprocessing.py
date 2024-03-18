import torch
from torch.utils.data import Dataset

class ShapeDataset(Dataset):
    def __init__(self, labels, samples, transform=None, targetTransform=None):
        self.labels = labels
        self.samples = samples
        self.transform = transform
        self.targetTransform = targetTransform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.targetTransform is not None:
            label = self.targetTransform(label)
        return sample, label

class SavedShapesDataset(ShapeDataset):
    def __init__(self, lables, sampleDirectory, transform=None, targetTransform=None):
        super().__init__(labels=lables, samples=sampleDirectory, transform=transform,
                         targetTransform=targetTransform)
    
    def __getitem__(self, index):
        pass