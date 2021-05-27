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
    for epoch in range(params['n_epochs']):
        mean_loss = []
        metric_cnt = 0
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
            metric_cnt += 1
            if len(mean_loss) >= params['verbose_loss_every'] * len(dataloader) or i == len(dataloader) - 1:
                print(f"epoch {epoch}, loss {np.mean(mean_loss)}")
                mean_loss = []
            if metric_cnt >= params['verbose_metrics_every'] * len(dataloader) or i == 0 or i == len(dataloader) - 1:
                hit10 = 0
                ndcg10 = 0
            #     model.eval()
            #     for idx2 in range(len(val_dataset)):
            #         source, user, target, neg_samples = val_dataset[idx2]
            #         if source is None:
            #             continue
            #         items_to_predict = torch.cat((target, neg_samples), 1)
            #         items_prediction = model()
            #         (targets_prediction, negatives_prediction) = torch.split(items_prediction,
            #                                                                  [1, items_prediction.size(1) - 1],
            #                                                                  dim=1)
                metric_cnt = 0
                print(f"epoch {epoch}, hit@10 {hit10}, ndcg@10 {ndcg10}")
        print(sum(mean_loss) / len(mean_loss))