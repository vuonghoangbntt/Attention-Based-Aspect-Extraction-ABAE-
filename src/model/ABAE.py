import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .module import SelfAttention


class ABAE(torch.nn.Module):
    """
        The model described in the paper ``An Unsupervised Neural Attention Model for Aspect Extraction''
        by He, Ruidan and  Lee, Wee Sun  and  Ng, Hwee Tou  and  Dahlmeier, Daniel, ACL2017
        https://aclweb.org/anthology/papers/P/P17/P17-1036/
    """

    def __init__(self, wv_dim: int = 200, asp_count: int = 30,
                 ortho_reg: float = 0.1, maxlen: int = 201, init_aspects_matrix=None, init_embeddings_weight=None, device='cpu'):
        """
        Initializing the model
        :param wv_dim: word vector size
        :param asp_count: number of aspects
        :param ortho_reg: coefficient for tuning the ortho-regularizer's influence
        :param maxlen: sentence max length taken into account
        :param init_aspects_matrix: None or init. matrix for aspects
        """
        super(ABAE, self).__init__()
        self.wv_dim = wv_dim
        self.asp_count = asp_count
        self.ortho = ortho_reg
        self.maxlen = maxlen
        self.device = device

        # w2v_model = W2VEmbReader(os.path.join(path, 'word2vec'))
        self.embedding = nn.Embedding.from_pretrained(init_embeddings_weight, freeze=True)
        self.attention = SelfAttention(wv_dim)
        self.linear_transform = torch.nn.Linear(self.wv_dim, self.asp_count)
        # self.softmax_aspects = torch.nn.Softmax()
        self.aspects_embeddings = nn.Parameter(torch.empty(size=(wv_dim, asp_count)))

        if init_aspects_matrix is None:
            torch.nn.init.xavier_uniform(self.aspects_embeddings)
        else:
            self.aspects_embeddings.data = init_aspects_matrix.T

    def get_aspects_importances(self, text_embeddings, mask=None):
        """
            Takes embeddings of a sentence as input, returns attention weights
        """

        # compute attention scores, looking at text embeddings average
        attention_weights = self.attention(text_embeddings, mask)

        # multiplying text embeddings by attention scores -- and summing
        # (matmul: we sum every word embedding's coordinate with attention weights)
        weighted_text_emb = torch.matmul(attention_weights.unsqueeze(1),  # (batch, 1, sentence)
                                         text_embeddings  # (batch, sentence, wv_dim)
                                         ).squeeze(1)

        # encoding with a simple feed-forward layer (wv_dim) -> (aspects_count)
        raw_importances = self.linear_transform(weighted_text_emb)

        # computing 'aspects distribution in a sentence'
        aspects_importances = F.softmax(raw_importances, dim=1)

        return attention_weights, aspects_importances, weighted_text_emb

    def forward(self, text_tokens, negative_samples_tokens, input_mask=None, negative_mask=None):
        ## Get sentences embeddings
        text_embeddings = self.embedding(text_tokens)
        negative_samples_texts = self.embedding(negative_samples_tokens)  # (batch_size,m, maxlen, wv_dim)
        # negative samples are averaged
        averaged_negative_samples = torch.sum(negative_samples_texts * negative_mask.unsqueeze(3), dim=2) / torch.sum(
            negative_mask, dim=2).unsqueeze(2)
        # encoding: words embeddings -> sentence embedding, aspects importances
        _, aspects_importances, weighted_text_emb = self.get_aspects_importances(text_embeddings, input_mask)
        # decoding: aspects embeddings matrix, aspects_importances -> recovered sentence embedding
        recovered_emb = torch.matmul(self.aspects_embeddings, aspects_importances.unsqueeze(2)).squeeze()

        # loss
        reconstruction_triplet_loss = ABAE._reconstruction_loss(weighted_text_emb,
                                                                recovered_emb,
                                                                averaged_negative_samples)
        # print(reconstruction_triplet_loss, reconstruction_triplet_loss.size())
        max_margin = torch.sum(reconstruction_triplet_loss, dim=1)
        # print(max_margin, max_margin.size())
        return self.ortho * self._ortho_regularizer() + max_margin

    @staticmethod
    def _reconstruction_loss(text_emb, recovered_emb, averaged_negative_emb):
        text_emb = F.normalize(text_emb, dim=-1)  # batch_size*emb
        recovered_emb = F.normalize(recovered_emb, dim=-1)  # batch_size*emb
        averaged_negative_emb = F.normalize(averaged_negative_emb, dim=-1)  # batch_size*n*emb
        positive_dot_products = torch.matmul(text_emb.unsqueeze(1), recovered_emb.unsqueeze(
            2)).squeeze()  # (batch, 1, wv_dim)*(batch, wv_dim, 1) -> (batch)
        negative_dot_products = torch.matmul(averaged_negative_emb, recovered_emb.unsqueeze(
            2)).squeeze()  # (batch, m, wv_dim)*(batch, wv_dim, 1) -> (batch, m)
        reconstruction_triplet_loss = 1 - positive_dot_products.unsqueeze(1) + negative_dot_products
        return torch.max(reconstruction_triplet_loss, torch.zeros_like(reconstruction_triplet_loss))

    def _ortho_regularizer(self):
        aspects = F.normalize(self.aspects_embeddings, dim=0)
        return torch.norm(
            torch.matmul(aspects.t(), aspects) \
            - torch.eye(self.asp_count).to(self.device))

    def get_aspect_words(self, vocab, topn=15):
        words = []

        # getting aspects embeddings
        aspects = self.aspects_embeddings.detach().cpu().numpy()
        embedding = self.embedding.weight.detach().cpu().numpy()
        aspects = aspects / np.linalg.norm(aspects, axis=0, keepdims=True)
        embedding = embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)
        # getting scalar products of word embeddings and aspect embeddings;
        # to obtain the ``probabilities'', one should also apply softmax
        # words_scores = w2v_model.wv.syn0.dot(aspects)
        words_scores = np.dot(embedding, aspects)
        for row in range(aspects.shape[1]):
            argmax_scalar_products = np.argsort(- words_scores[:, row])[:topn]
            # print([w2v_model.wv.index2word[i] for i in argmax_scalar_products])
            # print([w for w, dist in w2v_model.similar_by_vector(aspects.T[row])[:topn]])
            words.append([vocab[i] for i in argmax_scalar_products])

        return words

    def predict_sentences(self, texts, mask):
        emb = self.embedding(texts).to(self.device)  # b*max_len*emb
        _, aspects_importances, weighted_text_emb = self.get_aspects_importances(emb, mask)  # b*emb
        weighted_text_emb = weighted_text_emb / torch.linalg.norm(weighted_text_emb, dim=-1, keepdims=True)
        aspects = self.aspects_embeddings  # emb*asp_count
        aspects = aspects / torch.linalg.norm(aspects, dim=0, keepdims=True)
        scores = torch.matmul(weighted_text_emb, aspects)
        return scores
