import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import ABAE
import os
from tqdm import tqdm
import logging
import numpy as np
from utils import reset_logger


class ABAETrainer:
    def __init__(self, args, w2v, vocab, train_dataset=None, test_dataset=None):
        self.args = args
        self.w2v = w2v
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.vocab = vocab
        self.base_name = 'ABAE_vocab_size={}_lr={}_aspect={}_ortho={}'.format(self.args.vocab_size,
                                                                              self.args.learning_rate,
                                                                              self.args.aspect_count,
                                                                              self.args.ortho_reg)
        self.save_path = os.path.join(self.args.output_dir,
                                      'vocab_size={}_lr={}_aspect={}_ortho={}/{}'.format(self.args.vocab_size,
                                                                                         self.args.learning_rate,
                                                                                         self.args.aspect_count,
                                                                                         self.args.ortho_reg,
                                                                                         self.args.domain))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.root_logger = logging.getLogger()
        self.root_logger = logging.getLogger()
        self.root_logger.setLevel(logging.INFO)
        reset_logger(self.root_logger)
        log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        file_handler = logging.FileHandler("{0}/{1}.log".format(self.save_path, f'experiments'),
                                           mode='w')
        file_handler.setFormatter(log_formatter)
        self.root_logger.addHandler(file_handler)

        ### Create model
        emb_weight = torch.nn.init.kaiming_uniform_(
            torch.zeros((self.args.vocab_size + 3, self.args.embedding_size))).numpy()
        emb_weight = w2v.get_emb_matrix_given_vocab(vocab, emb_weight)

        self.model = ABAE(
            wv_dim=args.embedding_size, asp_count=args.aspect_count,
            ortho_reg=args.ortho_reg, maxlen=args.maxlen,
            init_aspects_matrix=w2v.get_aspect_matrix(args.aspect_count), init_embeddings_weight=emb_weight,
            device=self.args.device
        )
        self.model.to(self.args.device)

    def train(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        criterion = nn.MSELoss(reduction="sum")
        batches_per_epoch = int(len(self.train_dataset) / self.args.batch_size)
        min_loss = 10000
        for epoch in range(self.args.num_epochs):
            train_loss = 0.0
            self.root_logger.info("-----------------------------------")
            self.root_logger.info(f"Epoch {epoch}")
            self.root_logger.info("-----------------------------------")
            for iteration, (tokens, mask) in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                current_size = tokens.size(0)
                tokens, mask = tokens.to(self.args.device), mask.to(self.args.device)
                negative_indices = list(np.random.choice(len(self.train_dataset), size=current_size*self.args.negative_samples))
                negative_samples, negative_masks = self.train_dataset[negative_indices, :]
                negative_samples = negative_samples. \
                    view(current_size, self.args.negative_samples, -1).to(self.args.device)
                negative_masks = negative_masks. \
                    view(current_size, self.args.negative_samples, -1).to(self.args.device)
                loss = torch.mean(
                    self.model.forward(tokens, negative_samples, input_mask=mask, negative_mask=negative_masks))
                if loss.isnan():
                    break
                loss.backward()
                optimizer.step()
                train_loss += loss.item() / batches_per_epoch
            self.root_logger.info("Train loss: %.3f\t Min loss: %.3f" % (train_loss, min_loss))
            if train_loss < min_loss:
                min_loss = train_loss
                aspect_file = open(os.path.join(self.save_path, 'aspect.log'), 'wt', encoding='utf-8')
                self.save_model()
                for i, aspect in enumerate(self.model.get_aspect_words(list(self.vocab.keys()), topn=50)):
                    self.root_logger.info("Loss improved....")
                    self.root_logger.info(i + 1, " ".join([a for a in aspect]))
                    aspect_file.write("Aspect %d:\n" % (i + 1))
                    aspect_file.write("|".join(aspect) + '\n')

    def save_model(self):
        torch.save({
            'model': self.model.state_dict(),
            'args': self.args,
            'vocab': self.vocab
        }, os.path.join(self.save_path, "model.pt"))
