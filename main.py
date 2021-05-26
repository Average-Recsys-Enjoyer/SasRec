import yaml
import time

import torch
from torch.utils.data import DataLoader
import torch.optim
import numpy as np

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
    model_params = params['model']
    dataset = SequentialDataset(dataset_params['path'], dataset_params['max_len'], dataset_params['n_neg_samples'])
    dataloader = DataLoader(dataset, batch_size=dataset_params['batch_size'], num_workers=2, shuffle=True)
    model = SasRec(dataset.n_items, dataset_params['max_len'], embed_size=model_params['embed_size'],
                   shared_embed=model_params['shared_embed'], dropout=model_params['dropout'], b=model_params['num_blocks'])
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(params['n_epochs']):
        mean_loss = []
        start = time.time()
        for i, batch in enumerate(dataloader):
            optim.zero_grad()
            source, target, pad_mask, neg_samples = batch
            ignore = (~pad_mask).int()
            logits, pos_embed, neg_embed = model(source, target, neg_samples, pad_mask)
            pos_logits = torch.sum(logits * pos_embed, dim=2)
            neg_logits = torch.sum(logits[:, :, None, :] * neg_embed, dim=(2, 3))
            loss = torch.sum(-torch.log(torch.sigmoid(pos_logits) + 1e-24) * ignore - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * ignore) / ignore.sum() * dataset_params['batch_size']
            mean_loss.append(loss.item())
            if len(mean_loss) >= params['verbose_every'] * len(dataloader):
                print("time took", time.time() - start)
                print(np.mean(mean_loss))
                mean_loss = []
                hit10 = 0
                start = time.time()
                for val_user, val_batch in enumerate(dataset):
                    print(val_user, val_user not in dataset.valid_data)
                    if val_user not in dataset.valid_data:
                        continue
                    _, val_target, val_pad_mask, _ = val_batch
                    logits, _, _ = model(torch.tensor(val_target[None]), None, None, torch.tensor(val_pad_mask[None]))
                    predictions = torch.matmul(model.item_embed.weight, logits[0, -1])
                    _, top10 = torch.topk(predictions, 10)
                    hit10 += dataset.valid_data[val_user] in top10
                print("time took", time.time() - start)
                print("hit@10", hit10 / len(dataset))
                break
            loss.backward()
            optim.step()