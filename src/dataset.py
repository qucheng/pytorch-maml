import sys
import numpy as np
from PIL import Image
import os

import torch
import torch.utils.data as data

class FewShotDataset(data.Dataset):
    """
    Load image-label pairs from a task to pass to Torch DataLoader
    Tasks consist of data and labels split into train / val splits
    """

    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.transform = transform # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.root = self.task.root
        self.split = split
        self.img_ids = self.task.train_ids if self.split == 'train' else self.task.val_ids
        self.labels = self.task.train_labels if self.split == 'train' else self.task.val_labels

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

class Omniglot(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(Omniglot, self).__init__(*args, **kwargs)

    def load_image(self, idx):
        ''' Load image '''
        im = Image.open('{}/{}'.format(self.root, idx)).convert('RGB')
        im = im.resize((28,28), resample=Image.LANCZOS) # per Chelsea's implementation
        im = np.array(im, dtype=np.float32)
        return im

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        im = self.load_image(img_id)
        if self.transform is not None:
            im = self.transform(im)
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return im, label

class MNIST(data.Dataset):

    def __init__(self, *args, **kwargs):
        super(MNIST, self).__init__(*args, **kwargs)

    def load_image(self, idx):
        ''' Load image '''
        # NOTE: we use the PNG dataset because meta-learning results in an error
        # when using the bitmap dataset and PyTorch unpacker
        im = Image.open('{}/{}.png'.format(self.root, idx)).convert('RGB')
        im = np.array(im, dtype=np.float32)
        return im

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = self.load_image(img_id)
        if self.transform is not None:
            img = self.transform(img)
        target = self.labels[idx]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

def norm_img(data):
    data = data / 255.0
    data[:, 0, :, :] = (data[:, 0, :, :] - 0.4914) / 0.2023
    data[:, 1, :, :] = (data[:, 1, :, :] - 0.4822) / 0.1994
    data[:, 2, :, :] = (data[:, 2, :, :] - 0.4465) / 0.2010
    return data

class Cifar100(FewShotDataset):
    def __init__(self, *args, **kwargs):
        super(Cifar100, self).__init__(*args, **kwargs)
        train_dir = "/private/home/tinayujiang/map/one_image_net/data/cifar100/train/fine_classes"
        test_dir = "/private/home/tinayujiang/map/one_image_net/data/cifar100/test/fine_classes"
        self.train_data = []
        self.test_data = []
        for i in range(0, 100):
            tmp_data = np.load(os.path.join(train_dir, 'data_%d.npy' % i))
            tmp_data = norm_img(tmp_data)
            self.train_data.append(tmp_data)
            tmp_data = np.load(os.path.join(test_dir, 'data_%d.npy' % i))
            tmp_data = norm_img(tmp_data)
            self.test_data.append(tmp_data)
        self.train_size = self.train_data[0].shape[0]
        self.test_size = self.test_data[0].shape[0]
            #if subset_size is not None:
            #    ss = random.sample(range(self.data.shape[0]), subset_size)
            #    self.data = self.data[ss, :, :, :]

    #def __len__(self):
    #    return self.train_data[0].shape[0]

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        label = img_id / (self.train_size + self.test_size)
        cls_idx = img_id % (self.train_size + self.test_size)
        if (cls_idx < self.train_size):
            data = self.train_data[label][cls_idx]
        else:
            data = self.test_data[label][cls_idx - self.train_size]
        return data, self.labels[idx]
