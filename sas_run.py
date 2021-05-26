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
    model_params = params['model_sas']
    dataset = SequentialDataset(dataset_params['path'], dataset_params['max_len'], dataset_params['n_neg_samples'])
    dataloader = DataLoader(dataset, batch_size=dataset_params['batch_size'], num_workers=2, shuffle=True)
    print()
    val_dataloader = DataLoader(dataset, batch_size=dataset_params['batch_size'], num_workers=2, shuffle=False)
    val_target = dataset.get_val()
    model = SasRec(dataset.n_items, dataset_params['max_len'], embed_size=model_params['embed_size'],
                   shared_embed=model_params['shared_embed'], dropout=model_params['dropout'], b=model_params['num_blocks'])
    model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(params['n_epochs']):
        mean_loss = []
        metric_cnt = 0
        start = time.time()
        for i, batch in enumerate(dataloader):
            optim.zero_grad()
            source, target, pad_mask, neg_samples, _ = batch
            source, target, pad_mask, neg_samples = source.cuda(), target.cuda(), pad_mask.cuda(), neg_samples.cuda()
            ignore = (~pad_mask).int()
            logits, pos_embed, neg_embed = model(source, target, neg_samples, pad_mask)
            pos_logits = torch.sum(logits * pos_embed, dim=2)
            neg_logits = torch.sum(logits[:, :, None, :] * neg_embed, dim=(2, 3))
            loss = torch.sum(-torch.log(torch.sigmoid(pos_logits) + 1e-24) * ignore - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * ignore) / (ignore.sum() * dataset_params['batch_size'])
            mean_loss.append(loss.item())
            metric_cnt += 1
            if len(mean_loss) >= params['verbose_loss_every'] * len(dataloader) or i == len(dataloader) - 1:
                print(f"epoch {epoch}, batch {i}, loss {np.mean(mean_loss)}")
                mean_loss = []
            if metric_cnt >= params['verbose_metrics_every'] * len(dataloader) or i == 0 or i == len(dataloader) - 1:
                hit10 = 0
                for val_batch in val_dataloader:
                    _, val_source, val_pad_mask, _, user_batch = val_batch
                    logits, _, _ = model(val_source.cuda(), None, None, val_pad_mask.cuda())
                    predictions = torch.matmul(model.item_embed.weight, logits[:, -1].T)
                    _, top10 = torch.topk(predictions, 10, dim=0)
                    for j, user in enumerate(user_batch):
                      if user.item() not in val_target:
                        continue
                      hit10 += val_target[user.item()] in top10[:, j]
                print(f"epoch {epoch}, batch {i}, hit@10 {hit10 / len(val_target)}")
            loss.backward()
            optim.step()
