import yaml
import time

import torch
from torch.utils.data import DataLoader
import torch.optim
import numpy as np

from data import CaserDataset, CaserValDataset
from model import Caser


def collate_fn(data):
    return tuple(torch.cat(batch, 0) for batch in zip(*data))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open('./default_config.yaml') as config:
        params = yaml.load(config, Loader=yaml.FullLoader)
    dataset_params = params['dataset']
    model_params = params['model_sas']
    dataset = CaserDataset(dataset_params['path'], dataset_params['max_len'], dataset_params['n_neg_samples'])
    dataloader = DataLoader(dataset, batch_size=dataset_params['batch_size'], num_workers=12, shuffle=True, collate_fn=collate_fn)
    val_dataset = CaserValDataset(dataset)
    model = Caser(len(dataset), dataset.n_items, dataset_params['max_len'])
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), amsgrad=True)
    metric_cnt = 0
    for epoch in range(params['n_epochs']):
        mean_loss = []
        model.train()
        for idx, i in enumerate(dataloader):
            optim.zero_grad()
            source, user, target, neg_samples = i
            items_to_predict = torch.cat((target, neg_samples), 1)
            items_prediction = model(source.to(device), user.to(device), items_to_predict.to(device))
            # print(source.shape, user.shape, target.shape)
            (targets_prediction, negatives_prediction) = torch.split(items_prediction, [target.size(1), neg_samples.size(1)], dim=1)

            loss = torch.mean(-torch.log(torch.sigmoid(targets_prediction) + 1e-24) - torch.log(1 - torch.sigmoid(negatives_prediction) + 1e-24))
            mean_loss.append(loss.item())
            loss.backward()
            optim.step()
            if len(mean_loss) >= params['verbose_loss_every'] * len(dataloader) or i == len(dataloader) - 1:
                print(f"epoch {epoch}, loss {np.mean(mean_loss)}")
                mean_loss = []
        metric_cnt += 1
        if metric_cnt >= params['verbose_metrics_every'] * params['n_epochs'] or metric_cnt == params['n_epochs'] or epoch + 1 == params['n_epochs']:
            hit10 = 0
            ndcg10 = 0
            model.eval()
            for idx2 in range(len(val_dataset)):
                source, user, target, neg_samples = val_dataset[idx2]
                if source is None:
                    continue
                items_to_predict = torch.cat((target, neg_samples), 0)
                items_prediction = model(source.unsqueeze(0).to(device), user.unsqueeze(0).to(device),
                                                 items_to_predict.unsqueeze(0).to(device), for_pred=True)
                rank = torch.argsort(torch.argsort(items_prediction))[0]
                if rank > 90:
                    hit10 += 1
                    ndcg10 += 1 / np.log2(100 - rank.cpu() + 2)
                if idx2 % 10000 == 0:
                    print(f"now: {idx2}, all: {len(val_dataset)}")
            model.train()
            metric_cnt = 0
            sz = len(val_dataset.dataset.valid_data)
            print(f"epoch {epoch}, hit@10 {hit10 / sz}, ndcg@10 {ndcg10 / sz}")
        print(sum(mean_loss) / len(mean_loss))