import torch
import torch.nn as nn
import torch
import torch.nn.functional as F

class SasRec(nn.Module):
    def __init__(self, n_items, max_len, embed_size=50, shared_embed=True, dropout=0.1, b=2, device='cuda:0'):
        super().__init__()
        self.item_embed = nn.Embedding(n_items, embed_size)
        if shared_embed:
            self.item_embed_prediction = self.item_embed
        else:
            self.item_embed_prediction = nn.Embedding(n_items, embed_size)# nn.Parameter(torch.normal(0, 1, (n_items, embed_size)))
        self.pos_embed = nn.Parameter(torch.normal(0, 1, (max_len, embed_size)))
        self.subsequent_mask = (torch.triu(torch.ones((max_len, max_len))) == 0).to(device)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=1, dim_feedforward=embed_size,
                                                        dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=b)

    def forward(self, sequence, positive, negative, padding_mask):
        # s - input sequence of length n, each s_i \in [0, n_items)
        e = self.item_embed(sequence) + self.pos_embed # shape - batch_size, max_len, embed_size
        e = e.permute(1, 0, 2)
        #print(e.shape)
        #print(padding_mask.shape)
        logits = self.encoder_layer(e, self.subsequent_mask, padding_mask) # max_len, batch_size, embed_size
        pos_embed, neg_embed = None, None
        if positive is not None and negative is not None:
            pos_embed = self.item_embed(positive)
            neg_embed = self.item_embed(negative)
        return logits.permute(1, 0, 2), pos_embed, neg_embed
        #return pos_embed * logits torch.matmul(attn_output, self.item_embed_prediction.weight.T).permute(1, 2, 0) # batch_size, n_item, max_len


"""
TODO:
different n: 10, 50, 100, 200, 300, 400, 500, 600
user_embed at last layer

в dataset'е нужно падить/обрезать до n + 1, передавать маску в которой True на месте падинга 
"""

activation_getter = {'iden': lambda x: x, 'relu': F.relu, 'tanh': torch.tanh, 'sigm': torch.sigmoid}

class Caser(nn.Module):
    def __init__(self, num_users, num_items, model_args):
        super(Caser, self).__init__()
        self.args = model_args

        # init args
        L = self.args.L
        dims = self.args.d
        self.n_h = self.args.nh
        self.n_v = self.args.nv
        self.drop_ratio = self.args.drop
        self.ac_conv = activation_getter[self.args.ac_conv]
        self.ac_fc = activation_getter[self.args.ac_fc]

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, dims)
        self.item_embeddings = nn.Embedding(num_items, dims)

        # vertical conv layer
        self.conv_v = nn.Conv2d(1, self.n_v, (L, 1))

        # horizontal conv layer
        lengths = [i + 1 for i in range(L)]
        self.conv_h = nn.ModuleList([nn.Conv2d(1, self.n_h, (i, dims)) for i in lengths])

        # fully-connected layer
        self.fc1_dim_v = self.n_v * dims
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        # W1, b1 can be encoded with nn.Linear
        self.fc1 = nn.Linear(fc1_dim_in, dims)
        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
        self.W2 = nn.Embedding(num_items, dims+dims)
        self.b2 = nn.Embedding(num_items, 1)

        # dropout
        self.dropout = nn.Dropout(self.drop_ratio)

        # weight initialization
        self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

        self.cache_x = None

    def forward(self, seq_var, user_var, item_var, for_pred=False):
        # Embedding Look-up
        item_embs = self.item_embeddings(seq_var).unsqueeze(1)  # use unsqueeze() to get 4-D
        user_emb = self.user_embeddings(user_var).squeeze(1)

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(item_embs)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(item_embs).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 1)
        # apply dropout
        out = self.dropout(out)

        # fully-connected layer
        z = self.ac_fc(self.fc1(out))
        x = torch.cat([z, user_emb], 1)

        w2 = self.W2(item_var)
        b2 = self.b2(item_var)

        if for_pred:
            w2 = w2.squeeze()
            b2 = b2.squeeze()
            res = (x * w2).sum(1) + b2
        else:
            res = torch.baddbmm(b2, w2, x.unsqueeze(2)).squeeze()

        return res