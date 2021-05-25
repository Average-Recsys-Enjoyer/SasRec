import torch
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
    main()