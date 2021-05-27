from collections import defaultdict

import torch
import yaml

import numpy as np
from torch.utils.data import Dataset


class SequentialDataset(Dataset):
    def __init__(self, path, max_len, n_neg_samples, split):
        self.max_len = max_len
        self.split = split
        self.n_neg_samples = n_neg_samples
        user_actions = self.read_data(path)
        self.preprocess_data(user_actions)
        self.all_items = set(range(self.n_items))

    def read_data(self, path):
        user_actions = defaultdict(list)
        n_items = 0
        with open(path) as f:
            for line in f:
                user, item = map(int, line.strip().split())
                user_actions[user].append(item)
                n_items = max(n_items, item)

        self.n_items = n_items + 1
        return user_actions

    def preprocess_data(self, user_actions):
        delete_users = []
        for user, items in user_actions.items():
            if len(items) == 1:
                delete_users.append(user)
        for user in delete_users:
            user_actions.pop(user)

        valid_data = {}
        test_data = {}
        train_data = defaultdict(np.array)
        for user, (_, items) in enumerate(user_actions.items()):
            n_items = len(items)
            if n_items > 3:
                valid_data[user] = items[-2]
                test_data[user] = items[-1]
                train_data[user] = items[:-2]
            else:
                train_data[user] = items

        self.add_pad_and_cut(train_data)

        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

    def make_neg_samples(self, user, target_shape):
        neg_samples = list(self.all_items - set(self.train_data[user]))
        return np.random.choice(neg_samples, (target_shape, self.n_neg_samples))

    def add_pad_and_cut(self, train_data):
        for user, items in train_data.items():
            items = items[-self.max_len - 1:]
            n_pad = self.max_len + 1 - len(items)
            train_data[user] = np.pad(items, (n_pad, 0))

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, user):
        sentence = self.train_data[user]
        source, target = sentence[:-1], sentence[1:]
        source_pad_mask = source == 0
        target_pad_mask = target == 0
        if self.split == 'train':
          neg_samples = self.make_neg_samples(user, self.max_len)
        else:
          neg_samples = self.make_neg_samples(user, 100)
        return source, target, source_pad_mask, target_pad_mask, neg_samples, user
    
    def get_val(self):
      return self.valid_data


class CaserDataset(SequentialDataset):
    def __init__(self, path, max_len, n_neg_samples):
        super().__init__(path, max_len, n_neg_samples, None)

    def add_pad_and_cut(self, train_data):
        for user, items in train_data.items():
            items = items[-self.max_len - 1:]
            train_data[user] = items

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, now_user):
        sentence = self.train_data[now_user]
        source = []
        target = []
        for idx, i in enumerate(sentence[1:]):
            items = sentence[:idx+1]
            n_pad = self.max_len - len(items)
            items = np.pad(items, (n_pad, 0))
            source.append(items)
            target.append(i)
        source = torch.tensor(source)
        target = torch.tensor(target).unsqueeze(1)
        neg_samples = self.make_neg_samples(now_user, target.shape[0])
        neg_samples = torch.tensor(neg_samples)
        return source, torch.tensor([now_user]).unsqueeze(0).repeat(target.shape[0], 1), target, neg_samples


class CaserValDataset(Dataset):
    def __init__(self, train_dataset, negatives=100):
        self.dataset = train_dataset
        self.neg = negatives

    def __len__(self):
        return len(self.dataset.train_data)

    def __getitem__(self, now_user):
        if now_user not in self.dataset.valid_data:
            return None, None, None, None
        items = self.dataset.train_data[now_user][-self.dataset.max_len:]
        n_pad = self.dataset.max_len - len(items)
        items = np.pad(items, (n_pad, 0))
        sentence = torch.tensor(items)

        neg_samples = self.dataset.make_neg_samples(now_user, self.neg)

        neg_samples = torch.tensor(neg_samples)

        target = torch.tensor([self.dataset.valid_data[now_user]])
        return sentence, torch.tensor([now_user]), target, neg_samples.squeeze(1)
