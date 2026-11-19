import torch
import numpy as np
# import piexif
# import warnings
# warnings.filterwarnings("error", category=UserWarning)
import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from PIL import Image, ImageFile

def load_vegfru(root, train_transform, test_transform):
    Vegfru.init(root)
    query_dataset = Vegfru(root, 'query', test_transform)
    train_dataset = Vegfru(root, 'train', train_transform)
    retrieval_dataset = Vegfru(root, 'retrieval', test_transform)
    # print(len(query_dataset))
    # print(len(train_dataset))


    return query_dataset, train_dataset, retrieval_dataset


class Vegfru(Dataset):

    def __init__(self, root, mode, transform=None, loader=default_loader):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.is_train = mode == 'train'
        if mode == 'train':
            self.data = Vegfru.TRAIN_DATA
            self.targets = Vegfru.TRAIN_TARGETS
            self.train_text = Vegfru.TRAIN_TEXT
        elif mode == 'query':
            self.data = Vegfru.QUERY_DATA
            self.targets = Vegfru.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = Vegfru.RETRIEVAL_DATA
            self.targets = Vegfru.RETRIEVAL_TARGETS
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')




    @staticmethod
    def init(root):
        # This file use train data (do not combine train and val as a new train dataset).
        # images_train = pd.read_csv(os.path.join(root, 'VegFru/vegfru_train_text_list.txt'), sep=' ',
        #                            names=['filepath', 'target', 'text'])
        data = []
        with open(os.path.join(root, 'vegfru_train_text_list.txt'), 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 2)  # 只分割前两个空格，第三部分保留空格
                if len(parts) == 3:
                    path, label, description = parts
                    data.append([path, int(label), description])
        images_train = pd.DataFrame(data, columns=['filepath', 'target', 'text'])

        images_val = pd.read_csv(os.path.join(root, 'vegfru_val.txt'), sep=' ',
                                 names=['filepath', 'target'])
        images_test = pd.read_csv(os.path.join(root, 'vegfru_test.txt'), sep=' ',
                                  names=['filepath', 'target'])
        train_images = []
        test_images = []
        for i in range(len(images_train)):
            train_images.append(images_train['filepath'][i])
        for i in range(len(images_test)):
            test_images.append(images_test['filepath'][i])
        # print(images_train[:10])
        label_list_train = []
        img_id_train = []
        for i in range(len(images_train)):
            label_list_train.append(int(images_train['target'][i]) + 1)
            img_id_train.append(i + 1)
        train_text = []
        for i in range(len(images_train)):
            train_text.append(images_train['text'][i])

        # print(label_list_train[:10])
        images_train = []
        # print(len(train_images))
        for i in range(len(train_images)):
            images_train.append([img_id_train[i], train_images[i], label_list_train[i],train_text[i]])
        # print(images_train[:10])
        images_train = pd.DataFrame(images_train, columns=['img_id', 'filepath', 'target','text'])

        k = len(train_images)
        label_list_test = []
        img_id_test = []
        for i in range(len(test_images)):
            label_list_test.append(int(images_test['target'][i]) + 1)
            img_id_test.append(k + i + 1)
        images_test = []
        for i in range(len(test_images)):
            images_test.append([img_id_test[i], test_images[i], label_list_test[i]])
        # print(images_test[:10])
        images_test = pd.DataFrame(images_test, columns=['img_id', 'filepath', 'target'])

        train_data = images_train
        test_data = images_test

        # Split dataset
        Vegfru.QUERY_DATA = test_data['filepath'].to_numpy()
        Vegfru.QUERY_TARGETS = test_data['target'] - 1

        Vegfru.TRAIN_DATA = train_data['filepath'].to_numpy()
        Vegfru.TRAIN_TARGETS = train_data['target'] - 1
        Vegfru.TRAIN_TEXT = train_data['text']

        Vegfru.RETRIEVAL_DATA = train_data['filepath'].to_numpy()
        Vegfru.RETRIEVAL_TARGETS = train_data['target'] - 1

    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            img = Image.open(os.path.join(self.root, self.data[idx])).convert('RGB')
        except:
            img = Image.open(os.path.join(self.root, self.data[idx])).convert('RGBA').convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.is_train:
            return img, self.targets[idx], self.train_text[idx]
        else:
            return img, self.targets[idx], idx