import codecs
import os
import operator
from utils import is_number
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class Data(Dataset):
    def __init__(self, sentences, masks):
        self.sentences = sentences
        self.masks = masks
        self.length = self.sentences.size()[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.sentences[idx], self.masks[idx]


class DataProcessor:
    def __init__(self, args):
        self.args = args
        self.file_map = {
            'train': self.args.train_file,
            'dev': self.args.dev_file,
            'test': self.args.test_file
        }

    def _create_vocab(self):
        # assert domain in {'restaurant', 'beer'}
        source = os.path.join(self.args.data_path, self.args.domain + '/train.txt')

        total_words, unique_words = 0, 0
        word_freqs = {}

        fin = codecs.open(source, 'r', 'utf-8')
        for line in fin:
            words = line.split()
            if self.args.maxlen is not None and len(words) > self.args.maxlen:
                continue

            for w in words:
                if not is_number(w):
                    try:
                        word_freqs[w] += 1
                    except KeyError:
                        unique_words += 1
                        word_freqs[w] = 1
                    total_words += 1

        logger.info('   %i total words, %i unique words' % (total_words, unique_words))
        sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)

        vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
        index = len(vocab)
        for word, _ in sorted_word_freqs:
            vocab[word] = index
            index += 1
            if self.args.vocab_size > 0 and index > self.args.vocab_size + 2:
                break
        if self.args.vocab_size > 0:
            logger.info('  keep the top %i words' % self.args.vocab_size)

        # Write (vocab, frequency) to a txt file
        vocab_file = codecs.open(
            os.path.join(self.args.data_path, self.args.domain + '/vocab'), mode='w', encoding='utf8')
        sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1))
        for word, index in sorted_vocab:
            if index < 3:
                vocab_file.write(word + '\t' + str(0) + '\n')
                continue
            vocab_file.write(word + '\t' + str(word_freqs[word]) + '\n')
        vocab_file.close()

        return vocab

    def _padding_sentences(self, sentences):
        mask = []
        for i in range(len(sentences)):
            if self.args.maxlen > len(sentences[i]):
                sentences[i].extend([self.args.padding_token for i in range(self.args.maxlen - len(sentences[i]))])
            else:
                sentences[i] = sentences[i][:self.args.maxlen]
            mask.append([1 if j != self.args.padding_token else 0 for j in sentences[i]])
        return torch.LongTensor(sentences), torch.ByteTensor(mask)

    def read_dataset(self, vocab, mode='train'):
        # assert domain in {'restaurant', 'beer'}
        assert mode in {'train', 'dev', 'test'}

        source = os.path.join(self.args.data_path, self.args.domain + '/' + self.file_map[mode])
        num_hit, unk_hit, total = 0., 0., 0.
        data_x = []

        fin = codecs.open(source, 'r', 'utf-8')
        for line in fin:
            words = line.strip().split()
            if 0 < self.args.maxlen < len(words):
                continue

            indices = []
            for word in words:
                if is_number(word):
                    indices.append(vocab['<num>'])
                    num_hit += 1
                elif word in vocab:
                    indices.append(vocab[word])
                else:
                    indices.append(vocab['<unk>'])
                    unk_hit += 1
                total += 1

            data_x.append(indices)

        logger.info(
            '   <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100 * num_hit / total, 100 * unk_hit / total))
        return self._padding_sentences(data_x)

    def get_data(self):
        logger.info('Reading data from '+self.args.domain)
        logger.info(' Creating vocab ...')
        vocab = self._create_vocab()
        logger.info(' Reading dataset ...')
        logger.info('  train set')
        train_data, train_mask = self.read_dataset(vocab, 'train')
        logger.info('  test set')
        test_data, test_mask = self.read_dataset(vocab, 'test')
        return vocab, Data(train_data, train_mask), Data(test_data, test_mask)
