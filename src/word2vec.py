import os
import codecs
import gensim
from tqdm import tqdm
import logging
import numpy as np
import torch
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


# word2vec pretraining
class Sentence(object):
    def __init__(self, file_name: str):
        self.file_name = file_name

    def __iter__(self):
        for line in tqdm(codecs.open(self.file_name, 'r', 'utf-8')):
            yield line.strip().split()


def pretrain_word2vec(args):
    sentences = Sentence(os.path.join(args.data_path, args.domain+'/'+args.train_file))
    print('Pretrained word embeddings....')
    model = gensim.models.Word2Vec(sentences=sentences, vector_size=args.embedding_size, window=args.window_size,
                                   min_count=args.min_count, workers=args.num_workers)
    model.save(os.path.join(args.data_path, args.domain+'/'+args.word2vec_file))


class W2VEmbReader:

    def __init__(self, emb_path, emb_dim=None):

        logger.info(f'Loading embeddings from: {emb_path}')
        self.embeddings = {}
        emb_matrix = []

        self.model = gensim.models.Word2Vec.load(emb_path)
        self.emb_dim = emb_dim
        for word in self.model.wv.index_to_key:
            self.embeddings[word] = torch.FloatTensor(self.model.wv[word])
            emb_matrix.append(list(self.model.wv[word]))
        self.vector_size = len(self.embeddings)
        self.emb_matrix = np.asarray(emb_matrix)

        logger.info('  #vectors: %i, #dimensions: %i' % (self.vector_size, self.emb_dim))

    def get_emb_given_word(self, word):
        try:
            return torch.FloatTensor(self.embeddings[word])
        except KeyError:
            return None

    def get_emb_matrix_given_vocab(self, vocab, emb_matrix):
        counter = 0.
        for word, index in vocab.items():
            try:
                emb_matrix[index] = self.embeddings[word]
                counter += 1
            except:
                pass
        logger.info(
            '%i/%i word vectors initialized (hit rate: %.2f%%)' % (counter, len(vocab), 100 * counter / len(vocab)))
        # L2 normalization
        # **********************************
        norm_emb_matrix = emb_matrix / np.linalg.norm(emb_matrix, axis=-1, keepdims=True)
        return torch.FloatTensor(norm_emb_matrix)

    def get_aspect_matrix(self, n_clusters):
        km = KMeans(n_clusters=n_clusters)
        km.fit(self.emb_matrix)
        clusters = km.cluster_centers_

        # L2 normalization
        norm_aspect_matrix = clusters / np.linalg.norm(clusters, axis=-1, keepdims=True)
        return torch.FloatTensor(norm_aspect_matrix)

    def get_emb_dim(self):
        return self.emb_dim
