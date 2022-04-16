import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, wv_dim: int):
        super(SelfAttention, self).__init__()
        self.d = wv_dim
        self.M = nn.Parameter(torch.empty(size=(wv_dim, wv_dim)))
        nn.init.xavier_uniform_(self.M.data)

    def forward(self, input_embeddings, mask=None):
        """
        input_embeddings: input word vectors with dim = (b, maxlen, wv_dim)
        mask: input word mask dim = (b, maxlen)
        """
        y = (torch.sum(input_embeddings * mask.unsqueeze(2), dim=1) / torch.sum(mask, dim=1).unsqueeze(1)).unsqueeze(
            2)  # (b, wv, 1)
        product_1 = torch.matmul(self.M, y)  # (b, wv, 1)

        # (b, maxlen, wv) x (b, wv, 1) -> (b, maxlen, 1)
        product_2 = torch.matmul(input_embeddings, product_1).squeeze(2) * mask
        product_2 = torch.exp(product_2 - torch.max(product_2, dim=1, keepdim=True)[0]) * mask
        return product_2 / torch.sum(product_2, dim=1).unsqueeze(1)
