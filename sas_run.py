import yaml
import time

import torch
from torch.utils.data import DataLoader
import torch.optim
import numpy as np

from data import SequentialDataset
from model import SasRec

if __name__ == "__main__":
    with open('./default_config.yaml') as config:
        params = yaml.load(config, Loader=yaml.FullLoader)
    dataset_params = params['dataset']
    model_params = params['model_sas']
    dataset_train = SequentialDataset(dataset_params['path'], dataset_params['max_len'], dataset_params['n_neg_samples'], split='train')
    dataset_val = SequentialDataset(dataset_params['path'], dataset_params['max_len'], dataset_params['n_neg_samples'], split='val')
    dataloader = DataLoader(dataset_train, batch_size=dataset_params['batch_size'], num_workers=2, shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=dataset_params['batch_size'], num_workers=2, shuffle=False)
    val_target = dataset_val.get_val()
    model = SasRec(dataset_val.n_items, dataset_params['max_len'], embed_size=model_params['embed_size'],
                   shared_embed=model_params['shared_embed'], dropout=model_params['dropout'], b=model_params['num_blocks'])
    model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    if params['from_pretrain'] is not None:
        checkpoint = torch.load(params['from_pretrain'] )
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])

    for epoch in range(params['n_epochs']):
        mean_loss = []
        metric_cnt = 0
        for i, batch in enumerate(dataloader):
            optim.zero_grad()
            source, target, pad_mask, _, neg_samples, user_batch = batch
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
            if metric_cnt >= params['verbose_metrics_every'] * len(dataloader) or (i == 0 and epoch == 0) or i == len(dataloader) - 1:
                hit10 = 0
                ndcg = 0
                for val_batch in val_dataloader:
                    _, val_source, _, pad_mask, neg_samples, user_batch = val_batch
                    target_items = []
                    for j, user in enumerate(user_batch):
                        if user in val_target:
                          target_items.append(val_target[user.item()])
                        else:
                          target_items.append(0)
                    target_items = torch.Tensor(target_items).long()
                    neg_samples = torch.cat((target_items[:, None], neg_samples[:, :, -1]), 1)
                    logits, _, neg_predictions = model(val_source.cuda(), None, neg_samples.cuda(), pad_mask.cuda())
                    predictions = -torch.bmm(neg_predictions, logits[:, -1][:, :, None])[:, :, 0]
                    ranks = torch.argsort(torch.argsort(predictions))[:, 0]
                    for j, user in enumerate(user_batch):
                      if user.item() not in val_target:
                        continue
                      if ranks[j] < 10:
                        hit10 += 1
                        ndcg += 1 / np.log2(ranks[j].cpu() + 2)
                metric_cnt = 0
                print(f"epoch {epoch}, batch {i}, hit@10 {hit10 / len(val_target)}, ndcg {ndcg / len(val_target)}")
            loss.backward()
            optim.step()
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
        }, './state_dict.pt')
