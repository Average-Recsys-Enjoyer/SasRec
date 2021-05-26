from collections import defaultdict
import yaml

import numpy as np
from torch.utils.data import Dataset, DataLoader


class SequentialDataset(Dataset):
    def __init__(self, path, max_len, n_neg_samples):
        self.max_len = max_len
        self.n_neg_samples = n_neg_samples
        user_actions = self.read_data(path)
        self.preprocess_data(user_actions)
        self.make_neg_samples()

    def read_data(self, path):
        user_actions = defaultdict(list)
        n_items = 0
        with open(path) as f:
            for line in f:
                user, item = map(int, line.strip().split())
                user_actions[user].append(item)
                n_items = max(n_items, item)

        self.n_items = n_items
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

        for user, items in train_data.items():
            items = items[-self.max_len:]
            n_pad = self.max_len - len(items)
            train_data[user] = np.pad(items, (n_pad, 0))

        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

    def make_neg_samples(self):
        neg_samples = {}
        all_items = set(range(self.n_items + 1))
        for user, items in self.train_data.items():
            neg_samples[user] = list(all_items - set(items))
        self.neg_samples = neg_samples

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, user):
        sentence = self.train_data[user]
        source, target = sentence[:-1], sentence[1:]
        pad_mask = source == 0
        neg_samples = np.random.choice(self.neg_samples[user], (self.max_len, self.n_neg_samples))
        return source, target, pad_mask, neg_samples
