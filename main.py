import yaml

import torch

from data import SequentialDataset
from model import SasRec


def main():
    n_items = 5
    max_len = 10
    embed_size = 4
    model = SasRec(n_items, max_len, embed_size)
    input_ = torch.randint(0, n_items, (4, max_len))
    mask_padding = torch.ones((4, max_len)) == 0
    #print("input shape", input_.shape)
    #print("mask shape", mask_padding.shape)
    output = model(input_, mask_padding)
    #print("output shape", output.shape)


if __name__ == "__main__":
    #main()
    with open('./default_config.yaml') as config:
        params = yaml.load(config, Loader=yaml.FullLoader)
    dataset_params = params['dataset']
    dataset = SequentialDataset(dataset_params['path'], dataset_params['max_len'], dataset_params['n_neg_samples'])
    for example in dataset:
        source, target, pad_mask, neg_samples = example
        break