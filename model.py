import torch
import torch.nn as nn


class SasRec(nn.Module):
    def __init__(self, n_items, max_len, embed_size=50, shared_embed=True, dropout=0.1, b=2):
        super().__init__()
        self.item_embed = nn.Embedding(n_items, embed_size)
        if shared_embed:
            self.item_embed_prediction = self.item_embed
        else:
            self.item_embed_prediction = nn.Embedding(n_items, embed_size)# nn.Parameter(torch.normal(0, 1, (n_items, embed_size)))
        self.pos_embed = nn.Parameter(torch.normal(0, 1, (max_len, embed_size)))
        self.subsequent_mask = torch.triu(torch.ones((max_len, max_len))) == 0
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=1, dim_feedforward=embed_size,
                                                        dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=b)

    def forward(self, s, padding_mask):
        # s - input sequence of length n, each s_i \in [0, n_items)
        e = self.item_embed(s) + self.pos_embed # shape - batch_size, max_len, embed_size
        e = e.permute(1, 0, 2)
        attn_output = self.encoder_layer(e, self.subsequent_mask, padding_mask) # max_len, batch_size, embed_size
        return torch.matmul(attn_output, self.item_embed_prediction.weight.T).permute(1, 2, 0) # batch_size, n_item, max_len


"""
TODO:
different n: 10, 50, 100, 200, 300, 400, 500, 600
user_embed at last layer

в dataset'е нужно падить/обрезать до n + 1, передавать маску в которой True на месте падинга 
"""